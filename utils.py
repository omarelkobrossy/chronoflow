import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from tqdm import tqdm
import requests
import json
from scipy import stats
from math import exp
from sklearn.preprocessing import LabelEncoder
import numba as nb
import math
import time



def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Prepare a raw OHLCV + feature DataFrame for model training or inference.

    Steps performed:
        1. Normalises the datetime index column and sorts chronologically.
        2. Imputes missing values — expanding/rolling mean for numerics,
           forward/back fill for categoricals and datetimes.
        3. Converts boolean columns to integer (0/1).
        4. Replaces infinite values with imputed means.
        5. Label-encodes any remaining object columns.
        6. Separates the target variable(s) and base OHLCV columns from
           the returned feature list.

    Args:
        df: Raw DataFrame containing OHLCV columns, engineered features,
            and the target column ``Price_Change_5``.

    Returns:
        df:           Cleaned DataFrame with all columns ready for scaling.
        feature_cols: List of column names to use as model inputs.
        target_cols:  List containing the target column name(s).
    """
    print("\nStarting data preprocessing...")

    # Convert the index column to datetime if it exists
    if 'Unnamed: 0' in df.columns:
        df['Date'] = pd.to_datetime(df['Unnamed: 0'])
        df = df.drop('Unnamed: 0', axis=1)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date to ensure proper time series handling
    df = df.sort_values('Date')
    # Identify column types
    datetime_columns = df.select_dtypes(include=['datetime64']).columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    boolean_columns = df.select_dtypes(include=['bool']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    print(f"\nFound {len(numerical_columns)} numerical columns")
    print(f"Found {len(categorical_columns)} categorical columns")
    print(f"Found {len(datetime_columns)} datetime columns")

    # Remove Close_Original and datetime columns from numerical columns to avoid scaling them
    numerical_columns = numerical_columns.drop('Close_Original') if 'Close_Original' in numerical_columns else numerical_columns
    numerical_columns = numerical_columns.drop(datetime_columns, errors='ignore')

    print("\nHandling missing values...")
    # Handle missing values using appropriate methods for each data type
    for col in df.columns:
        if col in numerical_columns:
            # For numerical columns, use expanding mean for first 100 records, then rolling mean
            df[col] = df[col].fillna(
                df[col].expanding(min_periods=1).mean().shift(1)
            ).fillna(
                df[col].rolling(window=100, min_periods=1).mean().shift(1)
            )
        elif col in categorical_columns:
            # For categorical columns, use forward fill with backward fill as backup
            df[col] = df[col].ffill().bfill()
        elif col in datetime_columns:
            # For datetime columns, use forward fill with backward fill as backup
            df[col] = df[col].ffill().bfill()
        else:
            # For any other columns, use forward fill with backward fill as backup
            df[col] = df[col].ffill().bfill()

    print("Converting boolean features to numerical...")
    # Convert boolean features to numerical
    if len(boolean_columns) > 0:
        df[boolean_columns] = df[boolean_columns].astype(int)

    print("Handling infinite values...")
    # Handle infinite values in numerical columns only
    df[numerical_columns] = df[numerical_columns].replace([np.inf, -np.inf], np.nan)
    df[numerical_columns] = df[numerical_columns].fillna(
        df[numerical_columns].expanding(min_periods=1).mean().shift(1)
    ).fillna(
        df[numerical_columns].rolling(window=100, min_periods=1).mean().shift(1)
    )

    print("Encoding categorical variables...")
    # Encode categorical variables (fit on training data only)
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            # Create a combined set of unique values
            unique_values = df[col].astype(str).unique()
            
            # Create and fit the encoder with all possible values
            label_encoders[col] = LabelEncoder()
            label_encoders[col].fit(unique_values)
            
            # Transform the data using the same encoder
            df[col] = label_encoders[col].transform(df[col].astype(str))

    print("\nPreparing features and target...")
    # Prepare features and target
    target_cols = ['Price_Change_5']
    exclude_cols = target_cols + list(categorical_columns) + list(datetime_columns)

    # Add base OHLC columns to the exclude list
    base_ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Close_Original', 'Volume']
    exclude_cols.extend(base_ohlc_cols)

    # Remove Regime_Label from exclude_cols if present
    if 'Regime_Label' in exclude_cols:
        exclude_cols.remove('Regime_Label')

    # Also remove regime probability columns from exclude_cols if present
    regime_prob_cols = [col for col in df.columns if col.startswith('Regime_Prob_')]
    for col in regime_prob_cols:
        if col in exclude_cols:
            exclude_cols.remove(col)

    # Add Regime_Label and regime probability columns to feature_cols if present in the dataframe
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    if 'Regime_Label' in df.columns and 'Regime_Label' not in feature_cols:
        feature_cols.append('Regime_Label')
    for col in regime_prob_cols:
        if col not in feature_cols:
            feature_cols.append(col)

    print(f"Total number of features: {len(feature_cols)}")

    return df, feature_cols, target_cols

def _expanding_zscore_no_lookahead(df_num: pd.DataFrame) -> np.ndarray:
    """
    Vectorized expanding z-score with 1-step shift (no lookahead).
    Returns a float32 numpy array of shape (n_samples, n_features).
    """
    X = df_num.to_numpy(dtype=np.float64, copy=False)  # [N,F]
    # cumulative sums
    csum   = np.cumsum(np.nan_to_num(X, nan=0.0), axis=0)
    csum2  = np.cumsum(np.nan_to_num(X**2, nan=0.0), axis=0)
    n      = np.arange(1, X.shape[0] + 1, dtype=np.float64)[:, None]

    # expanding mean/std at t (including t). We need stats at t-1, so shift by 1.
    mean_t = csum / n
    var_t  = np.maximum(csum2 / n - mean_t**2, 0.0)
    std_t  = np.sqrt(var_t)

    # shift stats by 1 to avoid lookahead
    mean_tm1 = np.vstack([np.full((1, X.shape[1]), np.nan), mean_t[:-1]])
    std_tm1  = np.vstack([np.full((1, X.shape[1]), np.nan), std_t[:-1]])

    Z = (X - mean_tm1) / (std_tm1 + 1e-8)
    # For the first row (no history), z-scores are undefined → set to 0
    Z[0, :] = 0.0
    return Z.astype(np.float32, copy=False)

def make_ref_bins(x_ref: np.ndarray, bins: int = 20) -> np.ndarray:
    """
    Build a set of strictly-increasing quantile bin edges from a reference array.

    Edges are derived from the training slice only (no lookahead).  Duplicate
    quantile values are nudged by a small epsilon so every histogram bucket is
    non-empty by construction.

    Args:
        x_ref: 1-D reference array (typically the training window of one feature).
        bins:  Number of equal-probability bins (default 20).

    Returns:
        Array of ``bins + 1`` strictly-increasing edge values with ±inf caps.
    """
    # quantile edges from the *training* slice; strictly increasing
    q = np.quantile(x_ref[~np.isnan(x_ref)], np.linspace(0, 1, bins+1))
    q[0], q[-1] = -np.inf, np.inf
    # fix potential duplicates
    for i in range(1, len(q)):
        if q[i] <= q[i-1]:
            q[i] = q[i-1] + 1e-12
    return q

def psi_from_bins(
    x_ref: np.ndarray,
    x_cur: np.ndarray,
    edges: np.ndarray,
    eps: float = 1e-9,
) -> float:
    """
    Compute the Population Stability Index (PSI) between two distributions.

    PSI = Σ (ref_pct − cur_pct) × ln(ref_pct / cur_pct)

    Both histograms are built using the *training* bin edges so the comparison
    is anchored to the reference distribution's shape.  A PSI < 0.1 indicates
    stable feature distributions; > 0.2 suggests significant drift.

    Args:
        x_ref:  Reference (training) array for one feature.
        x_cur:  Current (live/validation) array for the same feature.
        edges:  Bin edges produced by :func:`make_ref_bins`.
        eps:    Floor applied to bin proportions to avoid log(0).

    Returns:
        Scalar PSI value (non-negative float).
    """
    # histogram using TRAIN edges for both ref and current
    r = np.histogram(x_ref[~np.isnan(x_ref)], edges)[0].astype(np.float64)
    c = np.histogram(x_cur[~np.isnan(x_cur)], edges)[0].astype(np.float64)
    r = r / max(r.sum(), 1.0)
    c = c / max(c.sum(), 1.0)
    r = np.clip(r, eps, None); c = np.clip(c, eps, None)
    return float(np.sum((r - c) * np.log(r / c)))

def hist_pdf(arr: np.ndarray, edges: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Compute a normalised histogram probability density from pre-computed edges.

    Args:
        arr:   Input array (NaN values are excluded).
        edges: Bin edges produced by :func:`make_ref_bins`.
        eps:   Minimum bin probability floor (avoids log(0) in PSI calcs).

    Returns:
        Array of length ``len(edges) - 1`` summing to ≈ 1.0.
    """
    h = np.histogram(arr[~np.isnan(arr)], edges)[0].astype(np.float64)
    s = h.sum()
    if s <= 0:
        return np.full_like(h, 1.0 / len(h), dtype=np.float64)
    return np.maximum(h / s, eps)

def psi_with_ref_pdf(
    cur_arr: np.ndarray,
    edges: np.ndarray,
    ref_pdf: np.ndarray,
    eps: float = 1e-9,
) -> float:
    """
    Compute PSI when the reference PDF is already pre-computed.

    Avoids re-histogramming the reference array on every call, which makes
    this the preferred variant for live inference loops where the reference
    distribution is cached once at training time.

    Args:
        cur_arr: Current feature array to compare against the reference.
        edges:   Bin edges from :func:`make_ref_bins` (training slice).
        ref_pdf: Pre-computed reference PDF from :func:`hist_pdf`.
        eps:     Minimum bin probability floor.

    Returns:
        Scalar PSI value (non-negative float).
    """
    cur_pdf = hist_pdf(cur_arr, edges, eps)
    # PSI = sum( (ref - cur) * log(ref/cur) )
    return float(np.sum((ref_pdf - cur_pdf) * np.log(ref_pdf / cur_pdf)))

def calculate_feature_importance(
    df: pd.DataFrame,
    feature_cols,
    target_cols,
    model_params=None,
    iterations: int = 1,
    save_importance: bool = False,
    visualize_importance: bool = False,
    K: int = 48,
    max_bin: int = 256
):
    """
    GPU-accelerated walk-forward feature importance (XGBoost total_gain).
    Picks top-K by average gain over 'iterations' expanding batches.
    """

    if model_params is None:
        raise ValueError("Pass your tuned XGBoost params (we'll add gpu settings for you).")

    # Ensure GPU params
    params = dict(model_params)  # shallow copy
    params.update({
        "n_estimators": 50,
        "max_depth": 4,
        "subsample": 0.5,
        "colsample_bytree": 0.5,
        "max_bin": max_bin,
        "device": "cuda",
        "tree_method": "hist"
    })

    n = len(df)
    iterations = max(1, int(iterations))
    batch_size = max(1, n // iterations)

    # Vectorized filtering using dtype masks
    available_cols = set(df.columns)
    requested_cols = set(feature_cols)
    
    # Find columns that exist in both requested features and dataframe
    valid_cols = list(requested_cols.intersection(available_cols))
    
    if not valid_cols:
        raise ValueError("No requested features found in dataframe")
    
    # Get numeric columns using vectorized dtype check
    numeric_mask = df[valid_cols].select_dtypes(include=[np.number]).columns
    feat_names = list(numeric_mask)
    
    # Report excluded columns
    excluded_cols = set(valid_cols) - set(feat_names)
    if excluded_cols:
        print(f"Excluded {len(excluded_cols)} non-numeric columns: {sorted(excluded_cols)}")
    
    if not feat_names:
        raise ValueError("No valid numeric features found for feature importance calculation")
    
    print(f"Using {len(feat_names)} numeric features for importance calculation")
    target_name = target_cols if isinstance(target_cols, str) else target_cols[0]

    # Accumulate GAIN sums (not lists) to save memory
    gain_sum = {f: 0.0 for f in feat_names}
    gain_cnt = {f: 0    for f in feat_names}

    print("\n[Feature Selection] Walk-forward batches on GPU...")
    for batch_start in tqdm(range(0, n, batch_size), desc="FS Progress"):
        batch_end = min(batch_start + batch_size, n)
        train_data = df.iloc[:batch_end]

        # Standardize with expanding stats (no lookahead) for this batch
        X_train_df = train_data[feat_names]
        Z_train = _expanding_zscore_no_lookahead(X_train_df)   # np.float32 [B,F]
        y_train = train_data[target_name].to_numpy(np.float32, copy=False)

        # Build GPU-quantized DMatrix with feature names
        dtrain = xgb.QuantileDMatrix(Z_train, label=y_train, feature_names=feat_names, max_bin=max_bin)

        # Train booster (native API; faster than sklearn wrapper)
        num_boost_round = params.get("n_estimators", 500)
        booster = xgb.train(
            {k: v for k, v in params.items() if k != "n_estimators"},
            dtrain,
            num_boost_round=num_boost_round
        )

        # total_gain importance (dict: {feature_name: gain})
        gain_scores = booster.get_score(importance_type="total_gain")

        # accumulate
        for f in feat_names:
            g = float(gain_scores.get(f, 0.0))
            gain_sum[f] += g
            gain_cnt[f] += 1

    # Average gain (if a feature never used in a batch, its gain stays 0 for that batch)
    avg_importance = {
        f: (gain_sum[f] / max(gain_cnt[f], 1)) for f in feat_names
    }

    feature_importance = (
        pd.DataFrame({"Feature": feat_names, "Importance": [avg_importance[f] for f in feat_names]})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    if save_importance:
        import os, json
        os.makedirs('Features', exist_ok=True)
        features_file = f'Features/{params.get("symbol","features")}.json'
        with open(features_file, 'w') as f:
            json.dump({row.Feature: float(row.Importance) for _, row in feature_importance.iterrows()}, f, indent=4)
        print(f"Saved feature gains to {features_file}")

    if visualize_importance:
        import matplotlib.pyplot as plt
        topn = min(20, len(feature_importance))
        plt.figure(figsize=(12, 6))
        plt.barh(feature_importance['Feature'][:topn][::-1], feature_importance['Importance'][:topn][::-1])
        plt.title('Top Feature Gains (avg over batches)')
        plt.xlabel('Average total_gain')
        plt.tight_layout()
        plt.savefig("DB/charts/feature_importance.png")
        print("Chart saved to DB/charts/feature_importance.png")

    # Top-K pick
    selected = feature_importance.head(K)['Feature'].tolist()
    print(f"\nSelected {len(selected)} features (top-K by avg total_gain)")
    return selected

def sanitize_features(df_train, feature_names, eps=1e-12):
    """Return a clean, ordered list of features that are present, numeric, non-constant in df_train."""
    cols = []
    for c in feature_names:
        if c not in df_train.columns:
            continue
        s = df_train[c]
        if not np.issubdtype(s.dtype, np.number):
            # try coercion
            s = pd.to_numeric(s, errors='coerce')
            if s.isna().all():
                continue
        # non-constant? (std over training window prior to scaling)
        if s.count() <= 1:
            continue
        if (s.max() - s.min()) < eps:
            continue
        cols.append(c)
    # preserve original order
    return cols

def standardize_expanding(train_df, hold_df, cols):
    """Expanding mean/std with shift(1). Returns float32 DataFrames with identical column order."""
    Xtr = train_df[cols].astype(np.float64).copy()
    Xho = hold_df[cols].astype(np.float64).copy()
    for c in cols:
        mean = Xtr[c].expanding(min_periods=1).mean().shift(1)
        std  = Xtr[c].expanding(min_periods=1).std().shift(1)
        Xtr[c] = (Xtr[c] - mean) / (std + 1e-8)
        last_mean = mean.iloc[-1]
        last_std  = std.iloc[-1] if not np.isnan(std.iloc[-1]) and std.iloc[-1] > 0 else 1.0
        Xho[c] = (Xho[c] - last_mean) / (last_std + 1e-8)
    return Xtr.astype(np.float32), Xho.astype(np.float32)

def standardize_expanding_train(train_df, cols):
    """Expanding mean/std with shift(1) on the training window only."""
    Xtr = train_df[cols].astype(np.float64).copy()
    for c in cols:
        mean = Xtr[c].expanding(min_periods=1).mean().shift(1)
        std  = Xtr[c].expanding(min_periods=1).std().shift(1)
        Xtr[c] = (Xtr[c] - mean) / (std + 1e-8)
    # first row had no history → NaNs → fill with 0
    Xtr = Xtr.fillna(0.0).astype(np.float32)
    return Xtr


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp *value* to the closed interval [min_value, max_value]."""
    return max(min_value, min(value, max_value))

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio from returns series"""
    excess_returns = returns - risk_free_rate
    return np.sqrt(365) * excess_returns.mean() / (excess_returns.std() + 1e-8)

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve"""
    rolling_max = equity_curve.expanding().max()
    drawdowns = equity_curve / rolling_max - 1
    return drawdowns.min()

def calculate_n_per_year(trade_count, start_date, end_date):
    """Calculate number of trades per year"""
    # Convert to pandas Timestamp if needed
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date)
    return trade_count / ((end_date - start_date).total_seconds() / (365.25 * 24 * 3600))

def calculate_cagr(initial_capital, final_capital, start_date, end_date):
    """
    Calculate Compound Annual Growth Rate (CAGR)
    
    Args:
        initial_capital: Starting capital amount
        final_capital: Ending capital amount
        start_date: Start date (pandas Timestamp or datetime)
        end_date: End date (pandas Timestamp or datetime)
    
    Returns:
        CAGR as a decimal (e.g., 0.15 for 15%)
    """
    # Convert to pandas Timestamp if needed
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.to_datetime(end_date)
    
    # Calculate number of years as float
    years = (end_date - start_date).total_seconds() / (365.25 * 24 * 3600)
    
    # Handle edge cases
    if years <= 0:
        return 0.0
    if final_capital <= 0:
        return -1.0  # Complete loss
    
    # Calculate CAGR: ((final / initial)^(1/years)) - 1
    cagr = ((final_capital / initial_capital) ** (1 / years)) - 1
    
    return cagr

def calculate_mar(cagr, max_drawdown, dd_min=0.01):
    """
    Calculate Managed Account Ratio (MAR)
    
    MAR = CAGR / |Max Drawdown|
    
    This ratio measures the return per unit of risk (drawdown).
    Higher MAR values indicate better risk-adjusted returns.
    
    Args:
        cagr: Compound Annual Growth Rate as decimal (e.g., 0.15 for 15%)
        max_drawdown: Maximum drawdown as decimal (e.g., -0.10 for -10%)
    
    Returns:
        MAR as a decimal (e.g., 1.5 for 1.5x)
    """
    
    # Convert max_drawdown to absolute value and calculate MAR
    mar = cagr / max(abs(max_drawdown), dd_min)
    
    return mar

def mar_signed_tanh(mar, s=50.0):
    """
    Calculate Managed Account Ratio (MAR) using tanh function to squash and scale the MAR value for optuna optimization
    """
    return float(np.tanh(mar / s))


def trade_factor_bandpass(
    n_per_year: float,
    lo: float = 800.0,
    hi: float = 2000.0,
    k: float = 200.0,
    floor: float = 0.1,
) -> float:
    """
    Sigmoid bandpass filter that rewards trade frequency within a target range.

    Returns a value in (floor, 1.0].  Strategies with annualised trade counts
    between *lo* and *hi* receive scores near 1.0; counts outside this window
    are penalised smoothly via logistic roll-off rather than a hard cutoff.

    Args:
        n_per_year: Annualised trade count for the candidate strategy.
        lo:         Lower target boundary (default 800 trades/year).
        hi:         Upper target boundary (default 2000 trades/year).
        k:          Sigmoid slope — smaller values widen the transition band.
        floor:      Minimum score returned (prevents zeroing out marginal trials).

    Returns:
        Scalar in [floor, 1.0].
    """
    s_lo = 1.0 / (1.0 + np.exp(-(n_per_year - lo) / k))
    s_hi = 1.0 / (1.0 + np.exp(-(hi - n_per_year) / k))
    tf = s_lo * s_hi
    return max(tf, floor)   # never 0; still distinguishes outside-band trials 

def win_edge_factor(win_rate_pct: float, rr: float, k: float = 6.0) -> float:
    """
    Compute a logistic edge factor relative to the breakeven win rate.

    The breakeven win rate for a given risk-reward ratio is 1 / (1 + RR).
    This function maps the distance between the actual win rate and that
    breakeven point onto (0, 1) via a logistic curve, making it suitable
    as a smooth multiplicative penalty/reward in composite objective functions.

    Args:
        win_rate_pct: Observed win rate as a percentage (e.g. 55.0 for 55%).
        rr:           Risk-reward ratio (target profit / stop loss distance).
        k:            Logistic sharpness — higher values steepen the curve.

    Returns:
        Scalar in (0, 1).  Returns 0.5 at the breakeven win rate.
    """
    w = win_rate_pct / 100.0
    w_be = 1.0 / (1.0 + rr)                   # breakeven win rate
    raw = (w - w_be) / max(1e-9, 1.0 - w_be)  # ≈ -1..1
    return 1.0 / (1.0 + np.exp(-k * raw))     # in (0,1), =0.5 at breakeven

def calculate_composite_score(mar, win_rate, rr, trade_count, start_date, end_date):
    """
    Calculate composite score that balances risk-adjusted returns and consistency
    
    Composite Score = MAR x win_rate_adj
    
    Where:
    - MAR = CAGR / |Max Drawdown| (risk-adjusted return ratio)
    - WinEdgeAdj = max(0, (win_rate - W_be) / (1 - W_be)) (adjusts win rate to break even win rate)
    
    This scoring system balances both risk-adjusted returns and consistency,
    ensuring that optimization favors strategies that not only have good returns 
    relative to their risk but also maintain reasonable win rates above 50%.
    
    Args:
        mar: Managed Account Ratio (CAGR / |Max Drawdown|)
        win_rate: Win rate as percentage (e.g., 60.0 for 60%)
        rr: Risk reward ratio
        trade_count: Number of trades
    Returns:
        Composite score (higher is better)
    """


    # Win rates above break even win rate get rewarded proportionally
    WinEdgeAdj = win_edge_factor(win_rate, rr)
    # Squash and scale the MAR value for optuna optimization
    mar_signed = mar_signed_tanh(mar, s=50.0)
    # Calculate Trade Factor.
    TradeFactor = trade_factor_bandpass(calculate_n_per_year(trade_count, start_date, end_date), k=250, lo=800, hi=2000)
    
    # Calculate composite score
    #composite_score = mar_signed * WinEdgeAdj * TradeFactor
    composite_score = mar
    
    return composite_score

def calculate_dynamic_slippage(entry_price):
    """Calculate slippage based on capital using logarithmic scaling"""
    min_slippage = 0.0001  # 0.01%
    max_slippage = 0.001   # 0.1%
    max_capital = 100000   # 100k
    
    # Use log scaling to calculate slippage
    # When capital = 0, slippage = min_slippage
    # When capital = max_capital, slippage = max_slippage
    # In between, it scales logarithmically
    if entry_price <= 0:
        return min_slippage
    
    # Calculate the log factor
    log_factor = np.log1p(entry_price) / np.log1p(max_capital)
    
    # Calculate the slippage
    slippage = min_slippage + (max_slippage - min_slippage) * log_factor
    
    return min(slippage, max_slippage)

def calculate_distribution_metrics(series):
    """Calculate distribution metrics for a series"""
    if series.nunique() <= 2:  # Skip binary or constant features
        return None

    try:
        mean = series.mean()
        std = series.std()
        skew = stats.skew(series)
        kurtosis = stats.kurtosis(series)
        median = series.median()
        min_ = series.min()
        max_ = series.max()

        # 1. Scale-free location
        scale_free_location = (mean - min_) / (max_ - min_) if max_ != min_ else 0

        # 2. Signal-to-noise ratio
        signal_to_noise = mean / std if std != 0 else 0

        # 3. Robust center deviation
        robust_center = mean / median if median != 0 else 0

        # 4. Drift across bars (mean of log returns)
        if (series > 0).all():
            log_returns = np.log(series).diff().dropna()
            mean_log_returns = log_returns.mean()
        else:
            mean_log_returns = np.nan  # Not defined for non-positive values

        # Calculate VaR (5%)
        var_95 = np.percentile(series, 5)
        
        # Calculate autocorrelation lag-1
        autocorr = series.autocorr(lag=1)
        
        # Calculate trend slope using linear regression
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        
        # Calculate entropy
        # Discretize the series into 10 bins
        hist, bin_edges = np.histogram(series, bins=10, density=True)
        # Calculate PMF (Probability Mass Function)
        pmf = hist * np.diff(bin_edges)
        # Calculate entropy
        entropy = stats.entropy(pmf)
        
        # Calculate Hurst Exponent
        # Using R/S (Rescaled Range) method
        lags = range(2, len(series)//2)
        tau = []; laggedvar = []
        
        for lag in lags:
            # Calculate price changes
            price_changes = series.diff(lag).dropna()
            # Calculate variance of price changes
            var = price_changes.var()
            # Only include non-zero variances
            if var > 0:
                laggedvar.append(var)
                tau.append(lag)
        
        # Only calculate Hurst if we have enough valid points
        if len(tau) > 1 and len(laggedvar) > 1:
            try:
                # Linear fit to double-log graph (log(tau) vs log(var))
                m = np.polyfit(np.log(tau), np.log(laggedvar), 1)
                hurst = m[0] / 2.0  # Hurst exponent is slope/2
            except:
                hurst = 0.5  # Default to random walk if calculation fails
        else:
            hurst = 0.5  # Default to random walk if not enough data points
        
        # In the result object, don't put underscores in the keys since in another function 
        # we split the result key from the feature's name by '_'
        result = {
            'mean': mean,
            'std': std,
            'skew': skew,
            'kurtosis': kurtosis,
            'median': median,
            'var95': var_95,
            'autocorrLag1': autocorr,
            'trendSlope': slope,
            'entropy': entropy,
            'hurst': hurst,
            'scaleFreeLocation': scale_free_location,
            'signalToNoise': signal_to_noise,
            'robustCenter': robust_center,
            #Removed meanLogReturns because it's not working as intended, Needs work
            #'meanLogReturns': mean_log_returns,
        }
        return result

    except Exception as e:
        return None
    
def analyze_window_features(df_window):
    """Analyze features in a window and return distribution metrics"""
    feature_metrics = {}
    
    # Get numerical columns
    numerical_cols = df_window.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        metrics = calculate_distribution_metrics(df_window[col])
        if metrics is not None:
            feature_metrics[col] = metrics
    
    return feature_metrics


def analyze_confidence_vs_success(trade_history_df, symbol):
    """
    Analyze the relationship between prediction confidence and trade success.
    Creates charts showing how trade performance varies with confidence levels.
    """
    if len(trade_history_df) == 0:
        print("No trades to analyze for confidence vs success relationship.")
        return
    
    # Create confidence categories based on predicted change magnitude
    trade_history_df['confidence_level'] = pd.cut(
        abs(trade_history_df['predicted_change']), 
        bins=5, 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Create risk percentage categories
    trade_history_df['risk_level'] = pd.cut(
        trade_history_df['risk_percentage'] * 100,
        bins=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Calculate metrics by confidence level
    confidence_analysis = trade_history_df.groupby('confidence_level').agg({
        'profit': ['count', 'mean', 'std'],
        'predicted_change': 'mean',
        'risk_percentage': 'mean'
    }).round(4)
    
    # Flatten column names
    confidence_analysis.columns = ['Trade_Count', 'Avg_Profit', 'Profit_Std', 'Avg_Predicted_Change', 'Avg_Risk_Pct']
    confidence_analysis['Win_Rate'] = trade_history_df.groupby('confidence_level')['profit'].apply(lambda x: (x > 0).mean() * 100).round(2)
    confidence_analysis['Profit_Per_Trade'] = confidence_analysis['Avg_Profit'] / confidence_analysis['Trade_Count']
    
    # Calculate metrics by risk level
    risk_analysis = trade_history_df.groupby('risk_level').agg({
        'profit': ['count', 'mean', 'std'],
        'predicted_change': 'mean',
        'risk_percentage': 'mean'
    }).round(4)
    
    risk_analysis.columns = ['Trade_Count', 'Avg_Profit', 'Profit_Std', 'Avg_Predicted_Change', 'Avg_Risk_Pct']
    risk_analysis['Win_Rate'] = trade_history_df.groupby('risk_level')['profit'].apply(lambda x: (x > 0).mean() * 100).round(2)
    risk_analysis['Profit_Per_Trade'] = risk_analysis['Avg_Profit'] / risk_analysis['Trade_Count']
    
    # Create visualization directory
    os.makedirs('DB/charts', exist_ok=True)
    
    # Create comprehensive confidence analysis chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{symbol} - Prediction Confidence vs Trade Success Analysis', fontsize=16, fontweight='bold')
    
    # 1. Win Rate by Confidence Level
    axes[0, 0].bar(confidence_analysis.index, confidence_analysis['Win_Rate'], 
                   color=['#ff7f7f', '#ffbf7f', '#ffff7f', '#bfff7f', '#7fff7f'])
    axes[0, 0].set_title('Win Rate by Prediction Confidence')
    axes[0, 0].set_ylabel('Win Rate (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Average Profit by Confidence Level
    colors = ['red' if x < 0 else 'green' for x in confidence_analysis['Avg_Profit']]
    axes[0, 1].bar(confidence_analysis.index, confidence_analysis['Avg_Profit'], color=colors)
    axes[0, 1].set_title('Average Profit by Prediction Confidence')
    axes[0, 1].set_ylabel('Average Profit ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 3. Trade Count by Confidence Level
    axes[0, 2].bar(confidence_analysis.index, confidence_analysis['Trade_Count'], 
                   color='skyblue', alpha=0.7)
    axes[0, 2].set_title('Trade Count by Prediction Confidence')
    axes[0, 2].set_ylabel('Number of Trades')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Win Rate by Risk Level
    axes[1, 0].bar(risk_analysis.index, risk_analysis['Win_Rate'], 
                   color=['#ff7f7f', '#ffbf7f', '#ffff7f', '#bfff7f', '#7fff7f'])
    axes[1, 0].set_title('Win Rate by Risk Level')
    axes[1, 0].set_ylabel('Win Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Average Profit by Risk Level
    colors = ['red' if x < 0 else 'green' for x in risk_analysis['Avg_Profit']]
    axes[1, 1].bar(risk_analysis.index, risk_analysis['Avg_Profit'], color=colors)
    axes[1, 1].set_title('Average Profit by Risk Level')
    axes[1, 1].set_ylabel('Average Profit ($)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 6. Scatter plot: Predicted Change vs Actual Profit
    scatter = axes[1, 2].scatter(trade_history_df['predicted_change'], trade_history_df['profit'], 
                                c=trade_history_df['risk_percentage'], cmap='viridis', alpha=0.6)
    axes[1, 2].set_title('Predicted Change vs Actual Profit')
    axes[1, 2].set_xlabel('Predicted Change')
    axes[1, 2].set_ylabel('Actual Profit ($)')
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 2], label='Risk Percentage')
    
    plt.tight_layout()
    plt.savefig(f'DB/charts/{symbol}_confidence_vs_success_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed correlation analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{symbol} - Detailed Confidence Analysis', fontsize=14, fontweight='bold')
    
    # Correlation heatmap
    correlation_data = trade_history_df[['predicted_change', 'risk_percentage', 'profit', 'size', 'trade_value']].corr()
    im = axes[0].imshow(correlation_data, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(correlation_data.columns)))
    axes[0].set_yticks(range(len(correlation_data.columns)))
    axes[0].set_xticklabels(correlation_data.columns, rotation=45)
    axes[0].set_yticklabels(correlation_data.columns)
    axes[0].set_title('Correlation Matrix')
    
    # Add correlation values to heatmap
    for i in range(len(correlation_data.columns)):
        for j in range(len(correlation_data.columns)):
            text = axes[0].text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
    
    # Profit distribution by confidence level
    confidence_levels = trade_history_df['confidence_level'].unique()
    colors = ['#ff7f7f', '#ffbf7f', '#ffff7f', '#bfff7f', '#7fff7f']
    
    for i, level in enumerate(confidence_levels):
        if pd.notna(level):
            level_data = trade_history_df[trade_history_df['confidence_level'] == level]['profit']
            axes[1].hist(level_data, alpha=0.6, label=f'{level}', color=colors[i], bins=20)
    
    axes[1].set_title('Profit Distribution by Confidence Level')
    axes[1].set_xlabel('Profit ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'DB/charts/{symbol}_detailed_confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"CONFIDENCE VS SUCCESS ANALYSIS - {symbol}")
    print(f"{'='*60}")
    
    print("\n📊 CONFIDENCE LEVEL ANALYSIS:")
    print("-" * 40)
    for level in confidence_analysis.index:
        if pd.notna(level):
            row = confidence_analysis.loc[level]
            print(f"{level:>10}: {row['Trade_Count']:>3} trades | "
                  f"Win Rate: {row['Win_Rate']:>5.1f}% | "
                  f"Avg Profit: ${row['Avg_Profit']:>7.2f} | "
                  f"Avg Predicted Change: {row['Avg_Predicted_Change']:>6.4f}")
    
    print("\n🎯 RISK LEVEL ANALYSIS:")
    print("-" * 40)
    for level in risk_analysis.index:
        if pd.notna(level):
            row = risk_analysis.loc[level]
            print(f"{level:>10}: {row['Trade_Count']:>3} trades | "
                  f"Win Rate: {row['Win_Rate']:>5.1f}% | "
                  f"Avg Profit: ${row['Avg_Profit']:>7.2f} | "
                  f"Avg Risk %: {row['Avg_Risk_Pct']*100:>5.1f}%")
    
    # Calculate key insights
    high_conf_trades = trade_history_df[trade_history_df['confidence_level'].isin(['High', 'Very High'])]
    low_conf_trades = trade_history_df[trade_history_df['confidence_level'].isin(['Low', 'Very Low'])]
    
    if len(high_conf_trades) > 0 and len(low_conf_trades) > 0:
        high_conf_win_rate = (high_conf_trades['profit'] > 0).mean() * 100
        low_conf_win_rate = (low_conf_trades['profit'] > 0).mean() * 100
        high_conf_avg_profit = high_conf_trades['profit'].mean()
        low_conf_avg_profit = low_conf_trades['profit'].mean()
        
        print(f"\n🔍 KEY INSIGHTS:")
        print("-" * 40)
        print(f"High Confidence Trades: {len(high_conf_trades):>3} trades | "
              f"Win Rate: {high_conf_win_rate:>5.1f}% | "
              f"Avg Profit: ${high_conf_avg_profit:>7.2f}")
        print(f"Low Confidence Trades:  {len(low_conf_trades):>3} trades | "
              f"Win Rate: {low_conf_win_rate:>5.1f}% | "
              f"Avg Profit: ${low_conf_avg_profit:>7.2f}")
        
        if high_conf_win_rate > low_conf_win_rate:
            print(f"✅ High confidence trades have {high_conf_win_rate - low_conf_win_rate:.1f}% higher win rate")
        else:
            print(f"❌ Low confidence trades have {low_conf_win_rate - high_conf_win_rate:.1f}% higher win rate")
            
        if high_conf_avg_profit > low_conf_avg_profit:
            print(f"✅ High confidence trades have ${high_conf_avg_profit - low_conf_avg_profit:.2f} higher average profit")
        else:
            print(f"❌ Low confidence trades have ${low_conf_avg_profit - high_conf_avg_profit:.2f} higher average profit")
    
    print(f"\n📈 Charts saved to:")
    print(f"   - DB/charts/{symbol}_confidence_vs_success_analysis.png")
    print(f"   - DB/charts/{symbol}_detailed_confidence_analysis.png")
    print(f"{'='*60}")

def send_telegram_message(message: str) -> dict:
    """
    Sends a notification message to a configured Telegram chat.

    Requires the following environment variables:
        TELEGRAM_BOT_TOKEN: The bot token issued by BotFather.
        TELEGRAM_CHAT_ID:   The target chat or user ID.

    Args:
        message: Plain-text message body to send.

    Returns:
        The Telegram API JSON response dict, or an error dict if
        credentials are not configured.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[Telegram] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping notification.")
        return {"ok": False, "description": "Credentials not configured"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload)
    return response.json()

def expected_slippage_bps(bar: dict, notional_Q: float, params) -> float:
    """
    Estimate expected transaction cost in basis points for a single order.

    The model combines three components:
        1. **Spread cost** — half-spread multiplied by taker probability.
        2. **Market impact** — square-root price impact scaled by participation rate.
        3. **Illiquidity adjustment** — Amihud illiquidity ratio modulates impact.

    Args:
        bar:        OHLCV bar dict with keys ``High``, ``Low``, ``Close``,
                    ``Volume``, and ``Return``.
        notional_Q: Order notional value in quote currency (USD / USDC).
        params:     Parameter namespace with attributes:
                        ``c_spr``      — spread coefficient,
                        ``taker_prob`` — probability of crossing the spread,
                        ``Y``          — market impact coefficient,
                        ``a``          — illiquidity scaling constant,
                        ``beta``       — illiquidity exponent.

    Returns:
        Estimated slippage in basis points (1 bps = 0.01%).
    """
    mid   = 0.5 * (bar['High'] + bar['Low'])
    Vdol  = bar['Close'] * bar['Volume'] + 1e-12
    sigma = (bar['High'] - bar['Low']) / max(mid, 1e-12)

    # Spread baseline (crossing); maker_prob in [0,1]
    spr_proxy = params.c_spr * sigma
    spr_baseline = (params.taker_prob) * spr_proxy  # 0 if pure maker

    # Participation rate
    phi = min(1.0, notional_Q / Vdol)

    # Square-root impact
    impact_bps = 1e4 * params.Y * sigma * (phi ** 0.5)

    # Liquidity scalers
    illiq = abs(bar['Return']) / Vdol
    S = (1.0 + params.a * (illiq ** params.beta))

    # Optional: volume percentile & time-of-day adjustments
    # S *= (1.0 + params.b * max(0.0, bar['VolPctMedian'] - 1.0))
    # S *= (1.0 + params.d * bar['TODPenalty'])

    return spr_baseline + impact_bps * S


# Binance order formatting helper functions
def get_filters(client, symbol):
    """Get trading filters for a symbol from Binance API"""
    info = client.exchange_info(symbol=symbol)
    sym = info["symbols"][0]
    f = {d["filterType"]: d for d in sym["filters"]}
    tick = float(f["PRICE_FILTER"]["tickSize"])
    step = float(f["LOT_SIZE"]["stepSize"])
    min_notional = float(f.get("MIN_NOTIONAL", {}).get("minNotional", 0))
    return tick, step, min_notional


def fmt_qty(qty, step):
    """Format quantity according to LOT_SIZE step size"""
    prec = max(0, int(round(-math.log10(step))))
    q = math.floor(qty * (10**prec)) / (10**prec)
    return f"{q:.{prec}f}".rstrip("0").rstrip(".")


def fmt_price(p, tick):
    """Format price according to PRICE_FILTER tick size"""
    prec = max(0, int(round(-math.log10(tick))))
    pr = round(p, prec)
    # Limit-maker must be on a valid tick
    return f"{pr:.{prec}f}".rstrip("0").rstrip(".")

def floor_to_step(x: float, step: float) -> float:
    """
    Floor *x* to the nearest multiple of *step* (exchange lot/tick granularity).

    Args:
        x:    Raw quantity or price value.
        step: Step size from the exchange's LOT_SIZE or PRICE_FILTER rule.

    Returns:
        Largest multiple of *step* that does not exceed *x*.
    """
    prec = max(0, int(round(-math.log10(step))))
    return math.floor(x * (10**prec)) / (10**prec)


                
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                return True, None
    
    return True, None