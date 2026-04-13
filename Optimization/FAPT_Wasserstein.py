import pandas as pd
import numpy as np
import json
import joblib
from typing import Dict, Any, List, Tuple
from scipy.stats import wasserstein_distance
# from FAPT_Metamodel_Feature_Importance import N_FEATURES_TO_SELECT

# Number of nearest historical windows to blend.
# Inverse-distance weights concentrate naturally on the closest regime,
# so this degrades gracefully toward nearest-neighbor when one window is clearly dominant.
K_NEAREST = 5

# Parameters that must be integers after interpolation (rounded, not floored).
DISCRETE_PARAMS = {
    'min_holding_period', 'max_holding_period', 'max_concurrent_trades',
    'n_estimators', 'max_depth', 'max_leaves', 'feature_count_k',
}

def get_top_market_weather_features(symbol: str) -> List[str]:
    """
    Get the top N_FEATURES_TO_SELECT features from the saved feature importance file.
    
    Args:
        symbol (str): The trading symbol
    
    Returns:
        List[str]: List of top feature names
    """
    features_path = f'Optimization/models/{symbol}_selected_features.json'
    try:
        with open(features_path, 'r') as f:
            feature_info = json.load(f)
            return feature_info['selected_features']
    except FileNotFoundError:
        raise FileNotFoundError(f"Feature importance file not found. Please run FAPT_Metamodel_Feature_Importance.py first.")

def calculate_feature_distances(current_features: Dict[str, float], window_features: Dict[str, Any], selected_features: List[str]) -> float:
    """
    Calculate the Wasserstein distance between current features and window features.
    
    Args:
        current_features (Dict[str, float]): Current window feature metrics
        window_features (Dict[str, Any]): Historical window feature metrics
        selected_features (List[str]): List of selected feature names to use
    
    Returns:
        float: Total Wasserstein distance across selected features
    """
    total_distance = 0.0
    
    for feature in selected_features:
        if feature in current_features and feature in window_features:
            # Create empirical distributions for comparison
            current_dist = np.array([current_features[feature]])
            window_dist = np.array([window_features[feature]])
            
            # Calculate Wasserstein distance for this feature
            distance = wasserstein_distance(current_dist, window_dist)
            total_distance += distance
    
    return total_distance

def predict_optimal_parameters(current_features: Dict[str, float], symbol: str = "TSLA") -> Dict[str, Any]:
    """
    Predict optimal parameters based on current feature metrics using the trained metamodel
    and Wasserstein distance for parameter selection.
    
    Args:
        current_features (Dict[str, float]): Dictionary containing feature metrics for the current window.
            Format should be: {
                'feature_name_mean': value,
                'feature_name_std': value,
                'feature_name_skew': value,
                'feature_name_kurtosis': value,
                ...
            }
        symbol (str): The trading symbol (default: "TSLA")
    
    Returns:
        Dict[str, Any]: Dictionary containing the predicted optimal parameters
    """
    # Get the top N features
    selected_features = get_top_market_weather_features(symbol)
    
    # Load the trained model and scaler
    model_path = f'Optimization/models/{symbol}_metamodel.joblib'
    scaler_path = f'Optimization/models/{symbol}_scaler.joblib'
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required model files not found. Please run FAPT_Metamodel_Feature_Importance.py first: {str(e)}")
    
    # Create a DataFrame with only the selected features
    feature_df = pd.DataFrame([{k: v for k, v in current_features.items() if k in selected_features}])
    
    # Ensure all required features are present
    missing_features = set(selected_features) - set(feature_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only the features used by the model
    X = feature_df[selected_features]
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    predicted_sharpe = model.predict(X_scaled)[0]
    
    # Load the optimization results to find the best parameters
    with open(f'Parameters/{symbol}_Aggregated_Optimization.json', 'r') as f:
        optimization_results = json.load(f)
    
    # ── Collect distances for every qualifying window ─────────────────────────
    window_distances: List[Tuple[float, Dict]] = []

    for window in optimization_results['results']:
        if 'feature_metrics' not in window:
            continue

        # Flatten nested feature_metrics into the same key format as current_features
        window_features: Dict[str, float] = {}
        for feature, metrics in window['feature_metrics'].items():
            for metric_name, value in metrics.items():
                feature_name = f"{feature}_{metric_name}"
                if feature_name in selected_features:
                    window_features[feature_name] = value

        # Skip windows that are missing any of the selected features
        if not all(f in window_features for f in selected_features):
            continue

        distance = calculate_feature_distances(current_features, window_features, selected_features)
        window_distances.append((distance, window))

    if not window_distances:
        raise ValueError("No matching parameters found in optimization results")

    # ── K-nearest selection ───────────────────────────────────────────────────
    window_distances.sort(key=lambda x: x[0])
    nearest_k = window_distances[:K_NEAREST]

    # ── Inverse-distance weights ──────────────────────────────────────────────
    # w_i = (1 / d_i) / Σ(1 / d_j)
    # If an exact match exists (d=0) it absorbs all weight, which is correct.
    inv_distances = [1.0 / d if d > 0 else float('inf') for d, _ in nearest_k]

    if any(w == float('inf') for w in inv_distances):
        # Exact match: give full weight to the zero-distance window(s)
        total = sum(1 for w in inv_distances if w == float('inf'))
        weights = [1.0 / total if w == float('inf') else 0.0 for w in inv_distances]
    else:
        total_inv = sum(inv_distances)
        weights = [w / total_inv for w in inv_distances]

    # ── Interpolate parameters across K nearest windows ───────────────────────
    # Gather the union of all parameter keys present in the K windows
    all_param_keys: set = set()
    for _, window in nearest_k:
        all_param_keys.update(window.get('best_parameters', {}).keys())

    interpolated: Dict[str, Any] = {}
    for key in all_param_keys:
        weighted_sum = 0.0
        weight_sum   = 0.0
        for (_, window), w in zip(nearest_k, weights):
            val = window.get('best_parameters', {}).get(key)
            if val is not None:
                weighted_sum += w * float(val)
                weight_sum   += w
        if weight_sum > 0:
            blended = weighted_sum / weight_sum
            # Round discrete parameters; leave continuous ones as floats
            interpolated[key] = round(blended) if key in DISCRETE_PARAMS else blended

    # ── Fill any missing parameters with safe defaults ────────────────────────
    required_parameters = {
        'min_risk_percentage': 0.005,
        'max_risk_percentage': 0.02,
        'risk_scaling_factor': 1.5,
        'risk_reward_ratio': 1.5,
        'min_predicted_move': 0.005,
        'partial_take_profit': 0.7,
        'min_holding_period': 5,
        'max_holding_period': 20,
        'max_concurrent_trades': 1,
        'window_fraction': 0.01,
        'retrain_fraction': 0.05
    }
    for param, default_value in required_parameters.items():
        if param not in interpolated:
            interpolated[param] = default_value

    # ── Build contribution metadata for inspection / logging ─────────────────
    contributing_windows = [
        {
            'window_id': window.get('window_id', 'unknown'),
            'wasserstein_distance': dist,
            'weight': w,
            'performance_metrics': window.get('performance_metrics', {}),
        }
        for (dist, window), w in zip(nearest_k, weights)
    ]

    return {
        'parameters': interpolated,
        'predicted_sharpe': predicted_sharpe,
        'wasserstein_distance': nearest_k[0][0],   # closest window distance
        'contributing_windows': contributing_windows,
        'k_nearest': len(nearest_k),
        'selected_features': selected_features,
    }

# Example usage:
if __name__ == "__main__":
    # Example feature metrics (you would get these from your live data)
    example_features = {
        '+DM_kurtosis': 2.5,
        'PV_Composite_std': 0.15,
        'Relative_Volume_MA_kurtosis': 3.2,
        'SPY_Direction_Align_MA_std': 0.12,
        '+DM_skew': 0.8,
        'Liquidity_Ratio_std': 0.25,
        'VXX_Beta_120_kurtosis': 2.8,
        'VP_Trend_SMA_5_skew': 0.6,
        'Volume_Momentum_skew': 0.4,
        'Regime_Prob_0_skew': 0.3,
        'QQQ_Beta_20_kurtosis': 2.6,
        'VXX_Corr_Regime_Change_std': 0.18,
        'VXX_Vol_Ratio_kurtosis': 2.9,
        'VP_Trend_mean': 0.45,
        'SPY_Beta_120_std': 0.22,
        'Upper_Shadow_Pct_kurtosis': 2.7,
        'VP_Trend_skew': 0.5,
        'Volume_Spike_Duration_kurtosis': 2.4,
        'SR_Break_Potential_kurtosis': 2.3,
        'QQQ_Rel_Strength_Change_skew': 0.35
    }
    
    try:
        optimal_params = predict_optimal_parameters(example_features)
        print("\nPredicted Optimal Parameters:")
        print(json.dumps(optimal_params, indent=4))
    except Exception as e:
        print(f"Error: {str(e)}")
