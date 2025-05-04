import pandas as pd
import numpy as np
import json
import joblib
from typing import Dict, Any, List
from scipy.stats import wasserstein_distance
# from FAPT_Metamodel_Feature_Importance import N_FEATURES_TO_SELECT

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
    
    # Find the window with the minimum Wasserstein distance
    best_window = None
    min_distance = float('inf')
    
    for window in optimization_results['results']:
        if 'feature_metrics' in window:
            # Extract only the selected features from the window
            window_features = {}
            for feature, metrics in window['feature_metrics'].items():
                for metric_name, value in metrics.items():
                    feature_name = f"{feature}_{metric_name}"
                    if feature_name in selected_features:
                        window_features[feature_name] = value
            
            # Only proceed if we have all required features
            if all(feature in window_features for feature in selected_features):
                distance = calculate_feature_distances(current_features, window_features, selected_features)
                
                if distance < min_distance:
                    min_distance = distance
                    best_window = window
    
    if best_window is None:
        raise ValueError("No matching parameters found in optimization results")
    
    # Extract parameters from the best window
    parameters = best_window.get('best_parameters', {})
    
    if not parameters:
        print("Warning: No parameters found in the best window. Window structure:", json.dumps(best_window, indent=2))
    
    # Return the parameters from the best matching window
    return {
        'parameters': parameters,
        'predicted_sharpe': predicted_sharpe,
        'wasserstein_distance': min_distance,
        'performance_metrics': best_window.get('performance_metrics', {}),
        'selected_features': selected_features,
        'window_id': best_window.get('window_id', 'unknown')
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
