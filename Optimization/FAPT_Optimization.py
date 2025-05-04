import pandas as pd
import numpy as np
from scipy import stats
import optuna
from Quant.TradingStrategy import run_strategy, calculate_sharpe_ratio, calculate_max_drawdown
from utils import preprocess_data
import json
from datetime import datetime
import os
import multiprocessing as mp
from functools import partial


# FAPT (Feature-Aggregation Parameter Tuning)
# This script performs a Bayesian optimization of the parameters for the trading strategy
# It uses a walk-forward analysis to optimize the parameters for each window of data which 
# is split by a step size and each window is of size WINDOW_SIZE
# It then aggregates the results of each window along with the feature metrics and saves them to a JSON file

# Total Bayesian Optimization trials = N_TRIALS_PER_STUDY * total_number_of_windows
# It is recommended to run this script using all the cores of the machine since it is a CPU intensive task




STEP_SIZE = 1000
N_TRIALS_PER_STUDY = 500

symbol = "TSLA"
data_path = f"DB/{symbol}_15min_indicators.csv"


def load_best_overall_window(symbol):
    with open(f'Parameters/{symbol}_Full_Optimization.json', 'r') as f:
        data = json.load(f)
    return data['top_parameters'][0]['parameters']['calculated_window_size']

WINDOW_SIZE = load_best_overall_window(symbol)

def calculate_distribution_metrics(series):
    """Calculate distribution metrics for a series"""
    if series.nunique() <= 2:  # Skip binary or constant features
        return None
    
    try:
        return {
            'mean': series.mean(),
            'std': series.std(),
            'skew': stats.skew(series),
            'kurtosis': stats.kurtosis(series)
        }
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

def optimize_parameters(df_window, feature_cols, target_cols, n_trials=N_TRIALS_PER_STUDY):
    """Run Bayesian optimization for a window"""
    def objective(trial):
        # Define parameter ranges
        min_risk = trial.suggest_float('min_risk_percentage', 0.005, 0.02)
        max_risk = trial.suggest_float('max_risk_percentage', min_risk, min(0.05, min_risk*1.5))
        scaling_factor = trial.suggest_float('risk_scaling_factor', 1.5, 3.0)
        reward_ratio = trial.suggest_float('risk_reward_ratio', 1.5, 3.0)
        min_predicted_move = trial.suggest_float('min_predicted_move', 0.005, 0.01)
        window_fraction = trial.suggest_float('window_fraction', 0.01, 0.5)
        retrain_fraction = trial.suggest_float('retrain_fraction', 0.05, 1)
        
        internal_window_size = int(len(df_window) * window_fraction)
        retrain_interval = max(int(internal_window_size * retrain_fraction), 10)
        
        # Run strategy with current parameters
        metrics = run_strategy(df_window, min_risk, max_risk, scaling_factor, 
                             reward_ratio, min_predicted_move, internal_window_size, retrain_interval,
                             feature_cols, target_cols)
        
        if metrics['trade_count'] == 0:
            return -1000.0
        
        # Store metrics as trial attributes
        trial.set_user_attr('total_return', metrics['total_return'])
        trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
        trial.set_user_attr('win_rate', metrics['win_rate'])
        trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
        trial.set_user_attr('trade_count', metrics['trade_count'])
        
        return metrics['sharpe_ratio']

    # Create and run optimization study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Get the best trial and its metrics
    best_trial = study.best_trial
    best_metrics = {
        'total_return': best_trial.user_attrs['total_return'],
        'sharpe_ratio': best_trial.user_attrs['sharpe_ratio'],
        'win_rate': best_trial.user_attrs['win_rate'],
        'max_drawdown': best_trial.user_attrs['max_drawdown'],
        'trade_count': best_trial.user_attrs['trade_count']
    }
    
    return {
        'parameters': study.best_params,
        'score': best_trial.value,
        'metrics': best_metrics
    }

def process_window(window_data):
    """Process a single window with its data"""
    window_df, window_idx, total_windows, feature_cols, target_cols = window_data
    
    print(f"\nProcessing window {window_idx + 1}/{total_windows}")
    print(f"Window period: {window_df['Date'].iloc[0]} to {window_df['Date'].iloc[-1]}")
    
    # Analyze features
    feature_metrics = analyze_window_features(window_df)
    
    # Optimize parameters
    optimization_results = optimize_parameters(window_df, feature_cols, target_cols)
    
    # Return results
    return {
        'window_start': window_df['Date'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
        'window_end': window_df['Date'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'feature_metrics': feature_metrics,
        'best_parameters': optimization_results['parameters'],
        'optimization_score': optimization_results['score'],
        'performance_metrics': optimization_results['metrics']
    }

def main():
    # Load data
    print(f"Loading data from {data_path}...")
    df, feature_cols, target_cols = preprocess_data(pd.read_csv(data_path))
    
    # Convert date column if needed
    if 'Unnamed: 0' in df.columns:
        df['Date'] = pd.to_datetime(df['Unnamed: 0'])
        df = df.drop('Unnamed: 0', axis=1)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Prepare window data for parallel processing
    total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE + 1
    print(f"Processing {total_windows} windows...")
    
    window_data = []
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        window_start = i
        window_end = i + WINDOW_SIZE
        window_df = df.iloc[window_start:window_end].copy()
        window_data.append((window_df, i//STEP_SIZE, total_windows, feature_cols, target_cols))
    
    # Determine number of processes to use
    num_processes = max(1, mp.cpu_count() - 1)  # Leave one core free
    print(f"Using {num_processes} processes for parallel optimization")
    
    # Process windows in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_window, window_data)
    
    # Save final results
    final_results = {
        'symbol': symbol,
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE,
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    os.makedirs('Parameters', exist_ok=True)
    with open(f'Parameters/{symbol}_Aggregated_Optimization.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    print("\nAnalysis complete! Results saved to Parameters directory.")

if __name__ == "__main__":
    main()
