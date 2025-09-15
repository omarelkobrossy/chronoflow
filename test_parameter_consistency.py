import pandas as pd
import numpy as np
from utils import preprocess_data, clamp
import sys
import os

# Add the current directory to the path so we can import from TradingStrategy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the run_strategy function
from TradingStrategy import run_strategy, DEFAULT_MIN_RISK, DEFAULT_MAX_RISK, DEFAULT_SCALING, DEFAULT_RR, DEFAULT_MIN_PREDICTED_MOVE, DEFAULT_WINDOW_SIZE, DEFAULT_RETREIN_INTERVAL, DEFAULT_PARTIAL_TAKE_PROFIT, DEFAULT_MIN_HOLDING_PERIOD, DEFAULT_MAX_HOLDING_PERIOD, DEFAULT_MAX_CONCURRENT_TRADES, DEFAULT_STOP_LOSS_ATR_MULTIPLIER, DEFAULT_ATR_PREDICTED_WEIGHT, MIN_WINDOW, MAX_WINDOW

def test_parameter_consistency():
    """Test that the same parameters give the same results in both modes"""
    
    print("=== PARAMETER CONSISTENCY TEST ===\n")
    
    # Load and prepare data (same as main script)
    symbol = "XRP_USD"
    data_path = f"DB/{symbol}_fifteenminute_indicators.csv"
    df, feature_cols, target_cols = preprocess_data(pd.read_csv(data_path))
    
    # Time filtering parameters
    start_date = '2025-01-15'
    end_date = '2025-03-31'
    
    # Use the new buffering logic (same as in the fixed main script)
    max_possible_window = MAX_WINDOW
    buffered_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=max_possible_window*15/60/24)
    df = df[(df['Date'] >= buffered_start_date) & (df['Date'] <= end_date)]
    
    print(f"Data length: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print()
    
    # Test 1: Use default parameters (simulating SKIP_OPTIMIZATION = True)
    print("TEST 1: Using DEFAULT parameters (simulating SKIP_OPTIMIZATION = True)")
    print(f"Window size: {DEFAULT_WINDOW_SIZE}")
    print(f"Retrain interval: {DEFAULT_RETREIN_INTERVAL}")
    
    metrics_default = run_strategy(df, 
                                 DEFAULT_MIN_RISK,
                                 DEFAULT_MAX_RISK, 
                                 DEFAULT_SCALING, 
                                 DEFAULT_RR, 
                                 DEFAULT_MIN_PREDICTED_MOVE, 
                                 DEFAULT_WINDOW_SIZE, 
                                 DEFAULT_RETREIN_INTERVAL,
                                 DEFAULT_PARTIAL_TAKE_PROFIT,
                                 DEFAULT_MIN_HOLDING_PERIOD,
                                 DEFAULT_MAX_HOLDING_PERIOD,
                                 DEFAULT_MAX_CONCURRENT_TRADES,
                                 feature_cols,
                                 target_cols,
                                 DEFAULT_STOP_LOSS_ATR_MULTIPLIER,
                                 DEFAULT_ATR_PREDICTED_WEIGHT)
    
    print(f"Default mode results:")
    print(f"  Total Return: {metrics_default['total_return']:.2f}%")
    print(f"  Final Capital: ${metrics_default['final_capital']:,.2f}")
    print(f"  Trade Count: {metrics_default['trade_count']}")
    print()
    
    # Test 2: Use same parameters but calculate window_size from window_fraction (simulating optimization)
    print("TEST 2: Using SAME parameters but calculated from window_fraction (simulating SKIP_OPTIMIZATION = False)")
    
    # Calculate what window_fraction would give us the same window_size
    window_fraction = DEFAULT_WINDOW_SIZE / len(df)
    calculated_window_size = clamp(int(len(df) * window_fraction), MIN_WINDOW, MAX_WINDOW)
    calculated_retrain_interval = max(int(calculated_window_size * (DEFAULT_RETREIN_INTERVAL / DEFAULT_WINDOW_SIZE)), 10)
    
    print(f"Window fraction: {window_fraction:.6f}")
    print(f"Calculated window size: {calculated_window_size}")
    print(f"Calculated retrain interval: {calculated_retrain_interval}")
    print(f"Original window size: {DEFAULT_WINDOW_SIZE}")
    print(f"Window sizes match: {'YES' if calculated_window_size == DEFAULT_WINDOW_SIZE else 'NO'}")
    
    metrics_optimization = run_strategy(df, 
                                      DEFAULT_MIN_RISK,
                                      DEFAULT_MAX_RISK, 
                                      DEFAULT_SCALING, 
                                      DEFAULT_RR, 
                                      DEFAULT_MIN_PREDICTED_MOVE, 
                                      calculated_window_size, 
                                      calculated_retrain_interval,
                                      DEFAULT_PARTIAL_TAKE_PROFIT,
                                      DEFAULT_MIN_HOLDING_PERIOD,
                                      DEFAULT_MAX_HOLDING_PERIOD,
                                      DEFAULT_MAX_CONCURRENT_TRADES,
                                      feature_cols,
                                      target_cols,
                                      DEFAULT_STOP_LOSS_ATR_MULTIPLIER,
                                      DEFAULT_ATR_PREDICTED_WEIGHT)
    
    print(f"Optimization mode results:")
    print(f"  Total Return: {metrics_optimization['total_return']:.2f}%")
    print(f"  Final Capital: ${metrics_optimization['final_capital']:,.2f}")
    print(f"  Trade Count: {metrics_optimization['trade_count']}")
    print()
    
    # Compare results
    print("COMPARISON:")
    return_diff = abs(metrics_default['total_return'] - metrics_optimization['total_return'])
    capital_diff = abs(metrics_default['final_capital'] - metrics_optimization['final_capital'])
    trade_diff = abs(metrics_default['trade_count'] - metrics_optimization['trade_count'])
    
    print(f"Return difference: {return_diff:.2f}%")
    print(f"Capital difference: ${capital_diff:,.2f}")
    print(f"Trade count difference: {trade_diff}")
    
    if return_diff < 0.01 and capital_diff < 1.0 and trade_diff == 0:
        print("✅ SUCCESS: Results are consistent between both modes!")
    else:
        print("❌ FAILURE: Results are different between modes!")
        print("This suggests there's still an issue with parameter consistency.")
    
    return metrics_default, metrics_optimization

if __name__ == "__main__":
    test_parameter_consistency()
