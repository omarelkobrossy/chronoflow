import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import os
from tqdm import tqdm
import shap # Import SHAP
import json
from scipy import stats

# Define model parameters for feature selection
model_params = {
    'n_estimators': 50,
    'max_depth': 4,
    'learning_rate': 0.1,
    'min_child_weight': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cpu'  # Changed from 'cuda' to 'cpu' for consistency
}

# Load the data
symbol = "TSLA"
input_dataset = f"DB/{symbol}_15min_indicators.csv"
df = pd.read_csv(input_dataset)
# df = df.iloc[:int(len(df) * 0.5)]

# Create train and holdout sets (75% train, 25% holdout)
# train_size = int(len(df) * 0.75)
# train_df = df.iloc[:train_size].copy()
# holdout_df = df.iloc[train_size:].copy()


# # Store original price for backtesting
# train_df['Close_Original'] = train_df['Close'].copy()
# holdout_df['Close_Original'] = holdout_df['Close'].copy()


def preprocess_data(df):
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


def calculate_feature_importance(df, feature_cols, target_cols, iterations=1, save_importance=False, visualize_importance=False, importance_threshold=0.01):
    feature_importance_scores = {}
    batch_size = len(df) // iterations
    
    # Calculate feature importance using expanding window
    print("\nImplementing walk-forward feature selection...")
    
    # Process in batches to better utilize GPU
    for batch_start in tqdm(range(0, len(df), batch_size), desc="Feature Selection Progress"):
        batch_end = min(batch_start + batch_size, len(df))
        
        # Get training data for the entire batch
        train_data = df.iloc[:batch_end].copy()  # Only use data up to current batch
        
        # Scale features using rolling statistics for the entire batch
        X_train = train_data[feature_cols].copy()
        for col in feature_cols:
            # Use expanding mean/std with shift to prevent look-ahead bias
            mean = X_train[col].expanding(min_periods=1).mean().shift(1)
            std = X_train[col].expanding(min_periods=1).std().shift(1)
            
            # Standardize
            X_train[col] = (X_train[col] - mean) / (std + 1e-8)
        
        # Convert to numpy arrays
        X_train_scaled = np.array(X_train.values, dtype=np.float32)
        y_train = np.array(train_data[target_cols].values.ravel(), dtype=np.float32)
        
        # Train model for feature importance
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Get feature importance for this batch
        for feature, importance in zip(feature_cols, model.feature_importances_):
            if feature not in feature_importance_scores:
                feature_importance_scores[feature] = []
            feature_importance_scores[feature].append(importance)

    # Calculate average importance scores
    avg_importance = {feature: np.mean(scores) for feature, scores in feature_importance_scores.items()}
    feature_importance = pd.DataFrame({
        'Feature': list(avg_importance.keys()),
        'Importance': list(avg_importance.values())
    }).sort_values('Importance', ascending=False)

    if save_importance:
        # Save selected features and their importance scores to JSON
        os.makedirs('Features', exist_ok=True)
        features_file = f'Features/{symbol}.json'
        features_to_save = {
            feature: float(importance)  # Convert numpy.float64 to Python float for JSON serialization
            for feature, importance in zip(feature_importance['Feature'], feature_importance['Importance'])
            if importance > importance_threshold  # Only include features above threshold
        }
        with open(features_file, 'w') as f:
            json.dump(features_to_save, f, indent=4)
            print(f"\nFeatures and importance scores saved to {features_file}")
            print(f"Saved {len(features_to_save)} features with importance > {importance_threshold}")
    
    if visualize_importance:
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.barh(feature_importance['Feature'][:20], feature_importance['Importance'][:20])
        plt.title('Top 20 Most Important Features (Expanding Window Analysis)')
        plt.xlabel('Average Importance Score')
        plt.tight_layout()
        plt.savefig("DB/charts/feature_importance.png")
        print("Feature importance chart saved to DB/charts/feature_importance.png")

    # Select top features based on average importance
    selected_features = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature'].tolist()
    
    # If less than 3 features meet the threshold, take the top 3 features
    if len(selected_features) < 3:
        selected_features = feature_importance['Feature'].head(3).tolist()
        print(f"\nWarning: Less than 3 features met the importance threshold. Using top 3 features instead.")
    
    print(f"\nSelected {len(selected_features)} features based on expanding window importance analysis")
    
    return selected_features


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio from returns series"""
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-8)

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve"""
    rolling_max = equity_curve.expanding().max()
    drawdowns = equity_curve / rolling_max - 1
    return drawdowns.min()

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
        # Calculate basic statistics
        mean = series.mean()
        std = series.std()
        skew = stats.skew(series)
        kurtosis = stats.kurtosis(series)
        median = series.median()
        
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
        
        result = {
            'mean': mean,
            'std': std,
            'skew': skew,
            'kurtosis': kurtosis,
            'median': median,
            'var95': var_95,
            'autocorr_lag1': autocorr,
            'trend_slope': slope,
            'entropy': entropy,
            'hurst': hurst
        }
        return result

    except:
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