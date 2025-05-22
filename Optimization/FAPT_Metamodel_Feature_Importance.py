import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# FAPT_Metamodel_Feature_Importance
# This script calculates the feature importance of the features used in the FAPT optimization
# It uses the results from the FAPT optimization to train a random forest regressor and then
# it calculates the feature importance of the features relating to the performance metrics, 
# selecing the most important features and plotting them



symbol = "TSLA"
N_FEATURES_TO_SELECT = 40

def load_optimization_results(symbol):
    """Load and process optimization results"""
    with open(f'Parameters/{symbol}_Aggregated_Optimization.json', 'r') as f:
        data = json.load(f)
    return data

def extract_feature_metrics(results):
    """Extract feature metrics and performance metrics from results"""
    feature_data = []
    performance_data = []
    
    for window_result in results['results']:
        # Extract performance metrics first
        if 'performance_metrics' in window_result:
            perf_metrics = {
                'sharpe_ratio': window_result['performance_metrics']['sharpe_ratio'],
                'total_return': window_result['performance_metrics']['total_return'],
                'win_rate': window_result['performance_metrics']['win_rate'],
                'max_drawdown': window_result['performance_metrics']['max_drawdown']
            }
            
            # Only extract feature metrics if we have performance metrics
            if 'feature_metrics' in window_result:
                window_features = {}
                for feature, metrics in window_result['feature_metrics'].items():
                    # Store all metrics for each feature
                    keys = list(metrics.keys())
                    for key in keys:
                        window_features[f"{feature}_{key}"] = metrics[key]
                
                
                feature_data.append(window_features)
                performance_data.append(perf_metrics)
    
    # Convert to DataFrames
    feature_df = pd.DataFrame(feature_data)
    performance_df = pd.DataFrame(performance_data)
    
    return feature_df, performance_df

def train_metamodel(feature_df, performance_df):
    """Train a model to predict Sharpe ratio based on feature metrics"""
    # Prepare features
    X = feature_df
    y = performance_df['sharpe_ratio']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_df.columns,
        'importance': model.feature_importances_
    })
    
    return model, feature_importance

def plot_feature_importance(feature_importance):
    """Plot top 20 features by importance"""
    plt.figure(figsize=(15, 8))
    top_features = feature_importance.nlargest(N_FEATURES_TO_SELECT, 'importance')
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top {N_FEATURES_TO_SELECT} Features by Importance')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'Optimization/{symbol}_feature_importance.png')
    plt.close()

def main():
    # Load results
    print("Loading optimization results...")
    results = load_optimization_results(symbol)
    
    # Extract data
    print("Extracting feature metrics...")
    feature_df, performance_df = extract_feature_metrics(results)
    
    # Train model
    print("Training metamodel...")
    model, feature_importance = train_metamodel(feature_df, performance_df)
    
    # Display results
    print(f"\nTop {N_FEATURES_TO_SELECT} Features by Importance:")
    top_features = feature_importance.nlargest(N_FEATURES_TO_SELECT, 'importance')
    print(top_features.to_string(index=False))
    
    # Plot results
    print("\nGenerating feature importance plot...")
    plot_feature_importance(feature_importance)
    print(f"Plot saved as 'Optimization/{symbol}_feature_importance.png'")

if __name__ == "__main__":
    main()
