import pandas as pd
import numpy as np
from utils import preprocess_data, clamp

# Load and prepare data (same as main script)
symbol = "XRP_USD"
data_path = f"DB/{symbol}_fifteenminute_indicators.csv"
df, feature_cols, target_cols = preprocess_data(pd.read_csv(data_path))

# Default parameters
DEFAULT_WINDOW_SIZE = 3233
DEFAULT_RETREIN_INTERVAL = 281

# Time filtering parameters
start_date = '2025-01-15'
end_date = '2025-03-31'

print("=== DATA FILTERING DEBUG ===\n")

# Original data info
print(f"Original data length: {len(df)}")
print(f"Original date range: {df['Date'].min()} to {df['Date'].max()}")

# Calculate buffered start date using DEFAULT_WINDOW_SIZE
buffered_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=DEFAULT_WINDOW_SIZE*15/60/24)
print(f"\nBuffered start date (using DEFAULT_WINDOW_SIZE={DEFAULT_WINDOW_SIZE}): {buffered_start_date}")

# Filter data using default parameters
df_default = df[(df['Date'] >= buffered_start_date) & (df['Date'] <= end_date)]
print(f"Default mode data length: {len(df_default)}")
print(f"Default mode date range: {df_default['Date'].min()} to {df_default['Date'].max()}")

# Now simulate optimization mode with different window sizes
print(f"\n=== OPTIMIZATION MODE SIMULATION ===")

# Simulate different window_fraction values that might be used in optimization
window_fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
MIN_WINDOW = 300
MAX_WINDOW = 50000

for window_fraction in window_fractions:
    # Calculate window_size as done in optimization
    window_size = clamp(int(len(df) * window_fraction), MIN_WINDOW, MAX_WINDOW)
    
    # Calculate what the buffered start date SHOULD be for this window_size
    correct_buffered_start = pd.to_datetime(start_date) - pd.Timedelta(days=window_size*15/60/24)
    
    # Filter data with correct buffered start date
    df_opt = df[(df['Date'] >= correct_buffered_start) & (df['Date'] <= end_date)]
    
    print(f"Window fraction: {window_fraction:.2f} -> Window size: {window_size}")
    print(f"  Correct buffered start: {correct_buffered_start}")
    print(f"  Data length: {len(df_opt)}")
    print(f"  Date range: {df_opt['Date'].min()} to {df_opt['Date'].max()}")
    print()

# Check if the issue is that optimization uses the wrong buffered start date
print("=== POTENTIAL ISSUE ANALYSIS ===")
print("The problem might be that in optimization mode:")
print("1. The buffered_start_date is calculated using DEFAULT_WINDOW_SIZE")
print("2. But the actual window_size used in optimization is different")
print("3. This means the data used for training might be insufficient or different")
print()
print("This could explain why you get different results with the same parameters!")

# Let's check what happens if we use the same parameters but different data
print("\n=== TESTING WITH SAME PARAMETERS BUT DIFFERENT DATA ===")

# Test case 1: Use default window size for buffering, but different actual window size
test_window_size = 1000  # Much smaller than DEFAULT_WINDOW_SIZE
test_buffered_start = pd.to_datetime(start_date) - pd.Timedelta(days=test_window_size*15/60/24)
df_test = df[(df['Date'] >= test_buffered_start) & (df['Date'] <= end_date)]

print(f"Test case: window_size={test_window_size}, but buffered with DEFAULT_WINDOW_SIZE={DEFAULT_WINDOW_SIZE}")
print(f"Data length: {len(df_test)}")
print(f"Date range: {df_test['Date'].min()} to {df_test['Date'].max()}")
print(f"Available data for training: {len(df_test) - test_window_size} bars")
print(f"Required data for training: {test_window_size} bars")
print(f"Sufficient data: {'YES' if len(df_test) >= test_window_size else 'NO'}")

if len(df_test) < test_window_size:
    print("❌ INSUFFICIENT DATA! This would cause the strategy to fail or behave unexpectedly.")
else:
    print("✅ Sufficient data available.")
