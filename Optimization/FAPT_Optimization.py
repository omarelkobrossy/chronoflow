import pandas as pd
import numpy as np
from scipy import stats
import optuna
import sys
import os
import argparse
import boto3
import json
from datetime import datetime
import multiprocessing as mp
from functools import partial

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TradingStrategy import run_strategy
from utils import clamp, preprocess_data, calculate_sharpe_ratio, calculate_max_drawdown, calculate_distribution_metrics, analyze_window_features


# FAPT (Feature-Aggregation Parameter Tuning)
# This script performs a Bayesian optimization of the parameters for the trading strategy
# It uses a walk-forward analysis to optimize the parameters for each window of data which 
# is split by a step size and each window is of size WINDOW_SIZE
# It then aggregates the results of each window along with the feature metrics and saves them to a JSON file

# Total Bayesian Optimization trials = N_TRIALS_PER_STUDY * total_number_of_windows
# It is recommended to run this script using all the cores of the machine since it is a CPU intensive task

STEP_SIZE = 310
N_TRIALS_PER_STUDY = 500
MIN_WINDOW = 300
MAX_WINDOW = 20000
symbol = "TSLA"
data_path = f"DB/{symbol}_15min_indicators.csv"
WINDOW_SIZE = 2350 #load_best_overall_window(symbol)


def load_best_overall_window(symbol):
    with open(f'Parameters/{symbol}_Full_Optimization.json', 'r') as f:
        data = json.load(f)
    return data['top_parameters'][0]['parameters']['calculated_window_size']



def optimize_parameters(df_window, feature_cols, target_cols, n_trials=N_TRIALS_PER_STUDY):
    """Run Bayesian optimization for a window"""
    def objective(trial):
        # Define parameter ranges
        min_risk = trial.suggest_float('min_risk_percentage', 0.005, 0.02)
        max_risk = trial.suggest_float('max_risk_percentage', min_risk, min(0.05, min_risk*1.5))
        scaling_factor = trial.suggest_float('risk_scaling_factor', 1.5, 3.0)
        reward_ratio = trial.suggest_float('risk_reward_ratio', 1.5, 3.0)
        min_predicted_move = trial.suggest_float('min_predicted_move', 0.005, 0.01)
        window_fraction = trial.suggest_float('window_fraction', 0.01, 0.5) # 1% to 50% of the data
        retrain_fraction = trial.suggest_float('retrain_fraction', 0.05, 1) # 5% to 100% of the window size
        window_size = clamp(int(len(df_window) * window_fraction), MIN_WINDOW, MAX_WINDOW)
        retrain_interval = max(int(window_size * retrain_fraction), 10)
        partial_take_profit = trial.suggest_float('partial_take_profit', 0.7, 0.95)
        min_holding_period = trial.suggest_int('min_holding_period', 5, 20)
        max_holding_period = trial.suggest_int('max_holding_period', min_holding_period, 40)
        max_concurrent_trades = trial.suggest_int('max_concurrent_trades', 1, 10)
        
        internal_window_size = int(len(df_window) * window_fraction)
        retrain_interval = max(int(internal_window_size * retrain_fraction), 10)
        
        # Run strategy with current parameters
        metrics = run_strategy(df_window, min_risk, max_risk, scaling_factor, 
                             reward_ratio, min_predicted_move, internal_window_size, retrain_interval,
                             partial_take_profit, min_holding_period, max_holding_period, max_concurrent_trades,
                             feature_cols, target_cols)
        
        if metrics['trade_count'] == 0:
            return -1000.0
        
        # Store metrics as trial attributes
        trial.set_user_attr('total_return', metrics['total_return'])
        trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
        trial.set_user_attr('win_rate', metrics['win_rate'])
        trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
        trial.set_user_attr('trade_count', metrics['trade_count'])
        
        return -abs(metrics['max_drawdown'])

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

def parse_args():
    parser = argparse.ArgumentParser(description='Run FAPT optimization with optional data subsetting')
    parser.add_argument('--start', type=int, help='Start index for data subsetting')
    parser.add_argument('--end', type=int, help='End index for data subsetting')
    parser.add_argument('--s3', action='store_true', help='Save results to S3 bucket')
    return parser.parse_args()

def save_to_s3(data, symbol, start_idx, end_idx):
    """Save results to S3 bucket"""
    s3 = boto3.client('s3')
    bucket_name = 'fapt-optimization-results'
    # Store in symbol-specific folder
    key = f"{symbol}/{symbol}_{start_idx}_{end_idx}.json"
    
    try:
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json.dumps(data, indent=4)
        )
        print(f"Successfully saved results to s3://{bucket_name}/{key}")
    except Exception as e:
        print(f"Error saving to S3: {str(e)}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load data
    print(f"Loading data from {data_path}...")
    df, feature_cols, target_cols = preprocess_data(pd.read_csv(data_path))

    # Subset data if start and end indices are provided
    if args.start is not None or args.end is not None:
        start_idx = args.start if args.start is not None else 0
        end_idx = args.end if args.end is not None else len(df)
        
        # Validate indices
        if start_idx < 0:
            print("Warning: Start index is negative, using 0 instead")
            start_idx = 0
        if end_idx > len(df):
            print(f"Warning: End index exceeds data length ({len(df)}), using data length instead")
            end_idx = len(df)
        if start_idx >= end_idx:
            raise ValueError("Start index must be less than end index")
            
        print(f"Subsetting data from index {start_idx} to {end_idx}")
        df = df.iloc[start_idx:end_idx]
        print(f"Subset data length: {len(df)}")
    
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
    num_processes = max(1, mp.cpu_count()) - 1
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
        'data_subset': {
            'start_index': args.start,
            'end_index': args.end,
            'subset_length': len(df)
        },
        'results': results
    }
    
    # Save locally
    os.makedirs('Parameters', exist_ok=True)
    with open(f'Parameters/{symbol}_Aggregated_Optimization.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    # Save to S3 if flag is set
    if args.s3:
        save_to_s3(final_results, symbol, args.start, args.end)
    
    print("\nAnalysis complete! Results saved to Parameters directory.")

if __name__ == "__main__":
    main()
