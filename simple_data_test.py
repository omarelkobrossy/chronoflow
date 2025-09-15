import pandas as pd
import numpy as np
from utils import preprocess_data
from TradingStrategy import MIN_WINDOW, MAX_WINDOW

def test_data_filtering():
    """Test that data filtering is now consistent"""
    
    print("=== DATA FILTERING CONSISTENCY TEST ===\n")
    
    # Load and prepare data
    symbol = "XRP_USD"
    data_path = f"DB/{symbol}_fifteenminute_indicators.csv"
    df, feature_cols, target_cols = preprocess_data(pd.read_csv(data_path))
    
    # Time filtering parameters
    start_date = '2025-01-15'
    end_date = '2025-03-31'
    
    # OLD METHOD (what was causing the issue)
    DEFAULT_WINDOW_SIZE = 3233
    old_buffered_start = pd.to_datetime(start_date) - pd.Timedelta(days=DEFAULT_WINDOW_SIZE*15/60/24)
    df_old = df[(df['Date'] >= old_buffered_start) & (df['Date'] <= end_date)]
    
    # NEW METHOD (fixed)
    max_possible_window = MAX_WINDOW
    new_buffered_start = pd.to_datetime(start_date) - pd.Timedelta(days=max_possible_window*15/60/24)
    df_new = df[(df['Date'] >= new_buffered_start) & (df['Date'] <= end_date)]
    
    print(f"Original data length: {len(df)}")
    print(f"Old method buffered start: {old_buffered_start}")
    print(f"Old method data length: {len(df_old)}")
    print(f"New method buffered start: {new_buffered_start}")
    print(f"New method data length: {len(df_new)}")
    print()
    
    # Test different window sizes that might be used in optimization
    test_window_sizes = [1000, 2000, 3233, 5000, 10000, 20000, 50000]
    
    print("Testing different window sizes:")
    print("Window Size | Old Method Sufficient | New Method Sufficient")
    print("------------|----------------------|----------------------")
    
    for window_size in test_window_sizes:
        old_sufficient = len(df_old) >= window_size
        new_sufficient = len(df_new) >= window_size
        
        print(f"{window_size:11d} | {'YES' if old_sufficient else 'NO':20s} | {'YES' if new_sufficient else 'NO':20s}")
    
    print()
    print("CONCLUSION:")
    if len(df_new) > len(df_old):
        print("✅ NEW METHOD PROVIDES MORE DATA")
        print("   This ensures optimization mode has enough data for any window size")
    else:
        print("❌ NEW METHOD DOESN'T PROVIDE MORE DATA")
        print("   This might still cause issues in optimization mode")
    
    # Check if we have enough data for the maximum possible window size
    if len(df_new) >= MAX_WINDOW:
        print("✅ SUFFICIENT DATA FOR MAXIMUM WINDOW SIZE")
    else:
        print("❌ INSUFFICIENT DATA FOR MAXIMUM WINDOW SIZE")
        print(f"   Need: {MAX_WINDOW} bars, Have: {len(df_new)} bars")

if __name__ == "__main__":
    test_data_filtering()
