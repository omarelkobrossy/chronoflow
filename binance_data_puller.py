import pandas as pd
from datetime import datetime
import logging
import os
import sys
from binance_bot import BinanceAPIClient, load_binance_api_key

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BinanceDataPuller:
    """
    A class to pull historical data from Binance API using BinanceAPIClient
    and save it in the same format as CoinbaseDataPuller.
    """
    
    def __init__(self, api_key=None, api_secret=None, symbol='XRP/USDC', years_of_data=3):
        """
        Initialize the Binance data puller.
        
        Args:
            api_key: Binance API key (optional, will load from file if not provided)
            api_secret: Binance API secret (optional, will load from file if not provided)
            symbol: Trading pair (default: 'XRP/USDC')
            years_of_data: Number of years of historical data to load (default: 2)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.years_of_data = years_of_data
        self.client = None
        
        # Load API keys from file if not provided
        if not self.api_key or not self.api_secret:
            self._load_api_keys()
        
        # Initialize the Binance API client if credentials are available
        if self.api_key and self.api_secret:
            try:
                self.client = BinanceAPIClient(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    symbol=self.symbol,
                    years_of_data=self.years_of_data
                )
                logger.info(f"Initialized Binance API client for {self.symbol}")
            except Exception as e:
                logger.error(f"Failed to initialize Binance client: {e}")
                self.client = None
        
        # Create data directory if it doesn't exist
        os.makedirs('DB', exist_ok=True)
    
    def _load_api_keys(self):
        """Load API keys from the Binance JSON file."""
        try:
            api_key, api_secret = load_binance_api_key()
            if api_key and api_secret:
                self.api_key = api_key
                self.api_secret = api_secret
                logger.info("Loaded API keys from settings/binance_api_key.json")
                logger.info(f"API Key: {self.api_key}")
                logger.info(f"API Secret: {self.api_secret[:20]}...")
            else:
                logger.warning("Failed to load API keys from file")
                logger.warning("Using public endpoints only")
                self.api_key = None
                self.api_secret = None
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            logger.warning("Using public endpoints only")
            self.api_key = None
            self.api_secret = None
    
    def pull_historical_data(self) -> pd.DataFrame:
        """
        Pull historical data from Binance API using BinanceAPIClient.
        
        Returns:
            DataFrame with historical data including technical indicators
        """
        if not self.client:
            logger.error("No authenticated client available. Please check your API credentials.")
            return pd.DataFrame()
        
        logger.info(f"Starting data pull for {self.symbol}")
        logger.info(f"Years of data: {self.years_of_data}")
        
        try:
            # Load historical data from API
            success = self.client.load_historical_data_from_api()
            
            if not success:
                logger.error("Failed to load historical data from API")
                return pd.DataFrame()
            
            # Get the historical data with indicators
            df = self.client.historical_data.copy()
            
            if df.empty:
                logger.warning("No data was fetched")
                return pd.DataFrame()
            
            logger.info(f"Data pull completed. Total rows fetched: {len(df)}")
            
            return df
        except Exception as e:
            logger.error(f"Error pulling historical data: {e}")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, product_id: str = None):
        """
        Save the data to a CSV file, formatting it similar to CoinbaseDataPuller's save_data.
        
        Args:
            df: DataFrame to save (should already have technical indicators)
            product_id: Product ID for filename (defaults to symbol)
        """
        if df.empty:
            logger.warning("No data to save")
            return
        
        # Use provided product_id or derive from symbol
        if product_id is None:
            # Convert symbol format (e.g., 'XRP/USDC' -> 'XRP_USD')
            product_id = self.symbol.replace('/', '_')
            # If it ends with USDC, convert to USD for consistency
            if product_id.endswith('_USDC'):
                product_id = product_id.replace('_USDC', '_USD')
        
        # Convert Date column to timestamp format (similar to CoinbaseDataPuller)
        df_formatted = df.copy()
        
        # Handle Date column or index - create timestamp and preserve Date column
        date_col = None
        if 'Date' in df_formatted.columns:
            # Convert Date to datetime if it's not already
            date_col = pd.to_datetime(df_formatted['Date'])
            # Format timestamp as 'YYYY-MM-DD HH:MM:SS'
            df_formatted['timestamp'] = date_col.dt.strftime('%Y-%m-%d %H:%M:%S')
            # Keep Date column but ensure it's date-only format (YYYY-MM-DD) as string
            df_formatted['Date'] = date_col.dt.strftime('%Y-%m-%d')
        elif isinstance(df_formatted.index, pd.DatetimeIndex):
            # Date is the index, reset it
            df_formatted = df_formatted.reset_index()
            # Check if index was named 'Date' or if we need to rename it
            if 'index' in df_formatted.columns:
                date_col = pd.to_datetime(df_formatted['index'])
                df_formatted = df_formatted.drop(columns=['index'])
            elif 'Date' in df_formatted.columns:
                date_col = pd.to_datetime(df_formatted['Date'])
            else:
                # Create Date from first datetime column
                datetime_cols = [col for col in df_formatted.columns if pd.api.types.is_datetime64_any_dtype(df_formatted[col])]
                if datetime_cols:
                    date_col = pd.to_datetime(df_formatted[datetime_cols[0]])
                    df_formatted = df_formatted.drop(columns=datetime_cols[0])
                else:
                    logger.warning("Could not determine Date column from index")
                    return
            
            # Format timestamp and Date columns
            df_formatted['timestamp'] = date_col.dt.strftime('%Y-%m-%d %H:%M:%S')
            df_formatted['Date'] = date_col.dt.strftime('%Y-%m-%d')
        else:
            # Try to find a datetime column
            datetime_cols = [col for col in df_formatted.columns if pd.api.types.is_datetime64_any_dtype(df_formatted[col])]
            if datetime_cols:
                date_col = pd.to_datetime(df_formatted[datetime_cols[0]])
                df_formatted['timestamp'] = date_col.dt.strftime('%Y-%m-%d %H:%M:%S')
                df_formatted['Date'] = date_col.dt.strftime('%Y-%m-%d')
                df_formatted = df_formatted.drop(columns=datetime_cols[0])
            else:
                logger.error("Could not find Date/timestamp column in DataFrame")
                logger.error(f"Available columns: {list(df_formatted.columns)}")
                return
        
        # Reorder columns to match Coinbase format: timestamp first, then other columns
        # Date should be positioned after ATR and before Time (if Time exists)
        # First, put timestamp first
        cols = ['timestamp']
        
        # Add all other columns except timestamp and Date
        other_cols = [col for col in df_formatted.columns if col not in ['timestamp', 'Date']]
        
        # Find the position to insert Date (after ATR, before Time)
        # Find the last ATR column (should be 'ATR' itself)
        atr_cols = [col for col in other_cols if 'ATR' in col]
        if atr_cols:
            # Find the last ATR column index in other_cols
            last_atr_idx = max([other_cols.index(col) for col in atr_cols])
            # Insert all columns up to and including the last ATR
            cols.extend(other_cols[:last_atr_idx + 1])
            # Insert Date after ATR
            cols.append('Date')
            # Insert remaining columns after Date
            cols.extend(other_cols[last_atr_idx + 1:])
        elif 'Time' in other_cols:
            # If Time exists but no ATR, insert Date before Time
            time_idx = other_cols.index('Time')
            cols.extend(other_cols[:time_idx])
            cols.append('Date')
            cols.extend(other_cols[time_idx:])
        else:
            # If neither ATR nor Time found, insert Date after basic OHLCV columns
            basic_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            last_basic_idx = -1
            for col in basic_cols:
                if col in other_cols:
                    last_basic_idx = max(last_basic_idx, other_cols.index(col))
            if last_basic_idx >= 0:
                cols.extend(other_cols[:last_basic_idx + 1])
                cols.append('Date')
                cols.extend(other_cols[last_basic_idx + 1:])
            else:
                # Fallback: insert Date after timestamp
                cols.append('Date')
                cols.extend(other_cols)
        
        df_formatted = df_formatted[cols]
        
        # Sort by timestamp
        df_formatted = df_formatted.sort_values('timestamp').reset_index(drop=True)
        
        # Create filename
        filename = f"DB/{product_id}_Binance.csv"
        
        # Save to CSV
        df_formatted.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
        logger.info(f"Data shape: {df_formatted.shape}")
        logger.info(f"Date range: {df_formatted['timestamp'].min()} to {df_formatted['timestamp'].max()}")
        
        return filename


def main():
    """Main function to demonstrate usage."""
    
    # Initialize the data puller (will automatically load API keys from file)
    puller = BinanceDataPuller(symbol='XRP/USDC', years_of_data=2)
    
    # Check if we have API credentials
    if not puller.client:
        logger.error("No API credentials available - cannot proceed with data pulling")
        logger.error("Please check your settings/binance_api_key.json file")
        return
    
    logger.info("Starting Binance historical data pull...")
    logger.info(f"Symbol: {puller.symbol}")
    logger.info(f"Years of data: {puller.years_of_data}")
    
    try:
        # Pull the data
        df = puller.pull_historical_data()
        
        if not df.empty:
            # Save the data
            filename = puller.save_data(df, product_id='XRP_USD')
            
            # Display summary
            print("\n" + "="*50)
            print("DATA PULL SUMMARY")
            print("="*50)
            print(f"Symbol: {puller.symbol}")
            print(f"Total rows: {len(df)}")
            print(f"Date range: {df['Date'].min() if 'Date' in df.columns else 'N/A'} to {df['Date'].max() if 'Date' in df.columns else 'N/A'}")
            print(f"Years of data: {puller.years_of_data}")
            print(f"Saved to: {filename}")
            print(f"Data shape: {df.shape}")
            print("="*50)
            
            # Show first few rows
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Show last few rows
            print("\nLast 5 rows:")
            print(df.tail())
            
        else:
            logger.error("No data was retrieved")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()

