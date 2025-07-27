# Retrieve 5m OHLC data on BTC/USD.
# Endpoint does not require authentication,
# but has utility functions for authentication.

import http.client
import urllib.request
import urllib.parse
import hashlib
import hmac
import base64
import json
import time
import pyperclip
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the Quant directory to the path to import from GatherData.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'Quant'))
from Quant.GatherData import calculate_technical_indicators

def main():
   # Read data from CSV file instead of calling Kraken API
   csv_path = r"C:\coding\Quant\DB\XRPUSD_15.csv"
   
   try:
      # Read the CSV file without headers and add them manually
      column_names = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Count']
      df = pd.read_csv(csv_path, header=None, names=column_names)
      
      # Convert timestamp column to datetime index
      df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
      df.set_index('timestamp', inplace=True)
      
      # Ensure columns are in the correct order and format
      df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
      
      # Convert all columns to float to ensure proper data types
      for col in df.columns:
         df[col] = df[col].astype(float)
      
      # Sort by timestamp to ensure chronological order
      df = df.sort_index()
      
      print(f"Loaded {len(df)} data points from {csv_path}")
      print(f"Date range: {df.index.min()} to {df.index.max()}")
      
      # Calculate technical indicators
      df_with_indicators = calculate_technical_indicators(df)
      
      # Save the results
      df_with_indicators.to_csv("XRP_USD_15min_indicators.csv")
      print(f"Saved {len(df_with_indicators)} data points to XRP_USD_15min_indicators.csv")
      
      # Print sample of the data
      print("\nFirst few rows of the data:")
      print(df_with_indicators.head())
      
      # Print date range
      print(f"\nDate range: {df_with_indicators.index.min()} to {df_with_indicators.index.max()}")
      
   except FileNotFoundError:
      print(f"Error: File not found at {csv_path}")
   except Exception as e:
      print(f"Error reading CSV file: {str(e)}")


def convert_kraken_data_to_dataframe(kraken_data):
    """
    Convert Kraken OHLC data to DataFrame format expected by calculate_technical_indicators
    
    Args:
        kraken_data: List of arrays from Kraken API
                    Each array: [time, open, high, low, close, vwap, volume, count]
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    # Convert the data to a list of dictionaries
    data_list = []
    
    for tick in kraken_data:
        # Kraken data format: [time, open, high, low, close, vwap, volume, count]
        timestamp, open_price, high_price, low_price, close_price, vwap, volume, count = tick
        
        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(timestamp)
        
        data_list.append({
            'Open': float(open_price),
            'High': float(high_price),
            'Low': float(low_price),
            'Close': float(close_price),
            'Volume': float(volume)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Create datetime index from timestamps
    timestamps = [tick[0] for tick in kraken_data]
    df.index = pd.to_datetime(timestamps, unit='s')
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_index()
    
    print(f"Converted {len(df)} data points from Kraken API")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


if __name__ == "__main__":
   main()