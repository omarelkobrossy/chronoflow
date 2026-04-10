import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os

def fetch_market_data(symbols=None, interval="15min", outputsize="full", years=15):
    """
    Fetch historical intraday data for market indices over a specified number of years
    and save to a consolidated DataFrame. Updates existing data if available.
    
    Args:
        symbols: List of market symbols (default=['SPY', 'QQQ', 'DIA', 'VIX'])
        interval: Time interval (default="15min")
        outputsize: "full" or "compact" (default="full")
        years: Number of years of historical data to fetch (default=15)
    
    Returns:
        Dictionary of DataFrames with market data
    """
    # Default market symbols if none provided
    if symbols is None:
        symbols = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'Nasdaq ETF',
            'DIA': 'Dow ETF',
            'VXX': 'VIX ETF'  # Changed from VIX to UVXY (ProShares Ultra VIX Short-Term Futures ETF)
        }
    
    # Alpha Vantage API configuration
    API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    BASE_URL = "https://www.alphavantage.co/query"
    
    # Define the path to save the consolidated market data
    MARKET_DATA_PATH = "DB/BroadMarket.csv"
    
    # Calculate current date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    # Load existing market data if available
    existing_market_data = None
    if os.path.exists(MARKET_DATA_PATH):
        print(f"Loading existing market data from {MARKET_DATA_PATH}")
        existing_market_data = pd.read_csv(MARKET_DATA_PATH)
        existing_market_data['timestamp'] = pd.to_datetime(existing_market_data['timestamp'])
        existing_market_data.set_index('timestamp', inplace=True)
    
    # Initialize dict to store market data
    market_data = {}
    consolidated_data = pd.DataFrame()
    
    # Process each market symbol
    for market_symbol, description in symbols.items():
        print(f"\nProcessing {market_symbol} ({description})...")
        
        # Special handling for VIX - rename it to ensure consistency in our code
        symbol_to_fetch = market_symbol
        
        # Determine date range to fetch
        if existing_market_data is not None:
            # Check if symbol exists in the data
            symbol_columns = [col for col in existing_market_data.columns if col.startswith(f"{market_symbol}_")]
            
            if len(symbol_columns) > 0:
                # Get the date range of existing data
                existing_start = existing_market_data.index.min()
                existing_end = existing_market_data.index.max()
                
                print(f"Found existing data for {market_symbol}: {existing_start} to {existing_end}")
                
                # Determine missing date ranges to fetch
                missing_ranges = []
                
                # Check if we need to fetch older data
                if existing_start > start_date:
                    missing_ranges.append(('old', start_date, existing_start - timedelta(days=1)))
                
                # Check if we need to fetch newer data
                if existing_end < end_date - timedelta(days=1):
                    missing_ranges.append(('new', existing_end + timedelta(days=1), end_date))
                
                if not missing_ranges:
                    print(f"No new data to fetch for {market_symbol}")
                    # Use existing data
                    symbol_data = existing_market_data[[col for col in existing_market_data.columns 
                                                   if col.startswith(f"{market_symbol}_")]]
                    # Rename columns to standard format
                    column_mapping = {
                        f"{market_symbol}_Open": "Open",
                        f"{market_symbol}_High": "High",
                        f"{market_symbol}_Low": "Low",
                        f"{market_symbol}_Close": "Close",
                        f"{market_symbol}_Volume": "Volume"
                    }
                    symbol_data = symbol_data.rename(columns={col: new_col 
                                                       for col, new_col in column_mapping.items() 
                                                       if col in symbol_data.columns})
                    
                    # For VIX, rename the key to 'VIX' for compatibility with existing code
                    if market_symbol == 'UVXY':
                        market_data['VIX'] = symbol_data
                    else:
                        market_data[market_symbol] = symbol_data
                    continue
            else:
                missing_ranges = [('all', start_date, end_date)]
        else:
            missing_ranges = [('all', start_date, end_date)]
        
        # Fetch missing data
        fetched_chunks = []
        for range_type, range_start, range_end in missing_ranges:
            print(f"Fetching {range_type} data for {market_symbol} from {range_start.date()} to {range_end.date()}")
            
            # Generate list of months to fetch
            current_date = range_start
            months_to_fetch = []
            
            while current_date <= range_end:
                months_to_fetch.append(current_date.strftime("%Y-%m"))
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
            
            # Track API calls to respect rate limit
            api_calls = 0
            last_reset_time = time.time()
            
            # Fetch data for each month
            for month in months_to_fetch:
                # Check if we need to pause to respect rate limit
                current_time = time.time()
                if api_calls >= 70:  # Leave some buffer below the 75 limit
                    elapsed = current_time - last_reset_time
                    if elapsed < 60:  # If less than a minute has passed
                        sleep_time = 60 - elapsed
                        print(f"Rate limit approaching. Pausing for {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                    api_calls = 0
                    last_reset_time = time.time()
                
                params = {
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": symbol_to_fetch,
                    "interval": interval,
                    "month": month,
                    "outputsize": outputsize,
                    "apikey": API_KEY
                }
                
                print(f"Fetching {market_symbol} data for {month}...")
                
                try:
                    response = requests.get(BASE_URL, params=params)
                    data = response.json()
                    
                    if f"Time Series ({interval})" in data:
                        month_df = pd.DataFrame.from_dict(data[f"Time Series ({interval})"], orient='index')
                        month_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        month_df = month_df.astype(float)
                        month_df.index = pd.to_datetime(month_df.index)
                        fetched_chunks.append(month_df)
                    else:
                        print(f"Warning: No data received for {market_symbol} - {month}. Response: {data}")
                    
                    api_calls += 1
                    time.sleep(0.8)  # Respect API limits
                    
                except Exception as e:
                    print(f"Error fetching {market_symbol} data for {month}: {str(e)}")
        
        if fetched_chunks:
            # Combine all fetched chunks
            fetched_df = pd.concat(fetched_chunks)
            fetched_df = fetched_df.sort_index()
            fetched_df = fetched_df[~fetched_df.index.duplicated(keep='first')]
            
            # If we have existing data, combine with fetched data
            if existing_market_data is not None and any(col.startswith(f"{market_symbol}_") for col in existing_market_data.columns):
                # Extract existing data for this symbol
                existing_symbol_data = pd.DataFrame(index=existing_market_data.index)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if f"{market_symbol}_{col}" in existing_market_data.columns:
                        existing_symbol_data[col] = existing_market_data[f"{market_symbol}_{col}"]
                
                # Combine with fetched data, avoiding duplicates
                combined_df = pd.concat([existing_symbol_data, fetched_df])
                combined_df = combined_df.sort_index()
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]  # Keep newer data if duplicated
                
                # Store in market data dict
                if market_symbol == 'UVXY':
                    market_data['VIX'] = combined_df
                else:
                    market_data[market_symbol] = combined_df
            else:
                # Just use fetched data
                if market_symbol == 'UVXY':
                    market_data['VIX'] = fetched_df
                else:
                    market_data[market_symbol] = fetched_df
        else:
            print(f"No new data fetched for {market_symbol}")
    
    # Create or update consolidated DataFrame
    if existing_market_data is not None:
        consolidated_data = existing_market_data.copy()
    
    # Update the consolidated DataFrame with new market data
    for original_symbol, df in market_data.items():
        # Determine the symbol to use in column names (use UVXY in storage for VIX data)
        storage_symbol = 'UVXY' if original_symbol == 'VIX' else original_symbol
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            consolidated_data[f"{storage_symbol}_{col}"] = df[col]
    
    # Ensure the index is named 'timestamp'
    consolidated_data.index.name = 'timestamp'
    
    # Save consolidated data
    print(f"\nSaving consolidated market data to {MARKET_DATA_PATH}")
    consolidated_data.to_csv(MARKET_DATA_PATH)
    
    # Clean up data for return
    for symbol in market_data:
        market_data[symbol] = market_data[symbol].sort_index()
    
    return market_data

def get_market_data_for_period(start_date, end_date, interval="15min"):
    """
    Retrieve market data for a specific period from the consolidated file.
    If data is missing, fetch it.
    
    Args:
        start_date: Start date for the period
        end_date: End date for the period
        interval: Time interval (default="15min")
    
    Returns:
        Dictionary of DataFrames with market data for the specified period
    """
    MARKET_DATA_PATH = "DB/BroadMarket.csv"
    
    # Convert dates to datetime if they're strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Check if market data file exists
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"Market data file not found. Fetching new data...")
        # Calculate years needed
        years_needed = (datetime.now() - start_date).days / 365 + 1
        return fetch_market_data(interval=interval, years=int(years_needed))
    
    # Load existing market data
    market_data = pd.read_csv(MARKET_DATA_PATH)
    market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
    market_data.set_index('timestamp', inplace=True)
    
    # Check if we have data for the entire period
    if market_data.index.min() <= start_date and market_data.index.max() >= end_date:
        print(f"Found complete market data for requested period")
        
        # Extract data for each symbol
        symbols = {'SPY', 'QQQ', 'DIA', 'UVXY'}
        result = {}
        
        for symbol in symbols:
            symbol_cols = [col for col in market_data.columns if col.startswith(f"{symbol}_")]
            if len(symbol_cols) == 5:  # We should have 5 columns (OHLCV) per symbol
                symbol_data = market_data.loc[start_date:end_date, symbol_cols].copy()
                
                # Rename columns to standard format
                column_mapping = {
                    f"{symbol}_Open": "Open",
                    f"{symbol}_High": "High",
                    f"{symbol}_Low": "Low",
                    f"{symbol}_Close": "Close",
                    f"{symbol}_Volume": "Volume"
                }
                
                # Rename only columns that exist
                rename_dict = {col: new_col for col, new_col in column_mapping.items() if col in symbol_data.columns}
                symbol_data = symbol_data.rename(columns=rename_dict)
                
                # Use VIX as the key for UVXY data to maintain compatibility with the rest of the code
                if symbol == 'UVXY':
                    result['VIX'] = symbol_data
                else:
                    result[symbol] = symbol_data
        
        return result
    else:
        print(f"Market data incomplete or missing for requested period. Fetching updated data...")
        # Calculate years needed to cover the requested period
        years_from_start = (datetime.now() - start_date).days / 365 + 1
        return fetch_market_data(interval=interval, years=int(years_from_start))

if __name__ == "__main__":
    try:
        # Fetch and update market data
        print("Fetching market data...")
        market_data = fetch_market_data(interval="15min", years=15)
        
        # Print information about the fetched data
        for symbol, df in market_data.items():
            print(f"\n{symbol} data summary:")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Total records: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")
            print(df.head(3))
        
        print("\nMarket data successfully updated and saved to DB/BroadMarket.csv")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}") 