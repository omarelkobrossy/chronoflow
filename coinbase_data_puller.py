import requests
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
import json
import os
from typing import List, Dict, Optional
import logging
from coinbase.rest import RESTClient
from GatherData import calculate_technical_indicators

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoinbaseDataPuller:
    """
    A class to pull historical data from Coinbase API using the official SDK
    with proper rate limiting and handling of the 350 bar limit per request.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 api_key_file: str = 'settings/cdp_api_key.json'):
        """
        Initialize the Coinbase data puller.
        
        Args:
            api_key: Coinbase API key (optional, will load from file if not provided)
            api_secret: Coinbase API secret (optional, will load from file if not provided)
            api_key_file: Path to API key file
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_key_file = api_key_file
        self.client = None
        
        # Load API keys from file if not provided
        if not self.api_key or not self.api_secret:
            self._load_api_keys()
        
        # Initialize the official Coinbase client if credentials are available
        if self.api_key and self.api_secret:
            try:
                self.client = RESTClient(api_key=self.api_key, api_secret=self.api_secret)
                logger.info("Initialized official Coinbase REST client")
            except Exception as e:
                logger.error(f"Failed to initialize Coinbase client: {e}")
                self.client = None
        
        # Rate limiting settings
        self.requests_per_second = 10  # Coinbase allows 10 requests per second
        self.min_interval = 1.0 / self.requests_per_second
        self.last_request_time = 0
        
        # Create data directory if it doesn't exist
        os.makedirs('DB', exist_ok=True)
    
    def _load_api_keys(self):
        """Load API keys from the specified JSON file."""
        try:
            if os.path.exists(self.api_key_file):
                with open(self.api_key_file, 'r') as f:
                    key_data = json.load(f)
                
                # Extract API key and secret from the JSON structure
                if 'name' in key_data and 'privateKey' in key_data:
                    # This is a CDP API key format
                    # The name field contains the API key name
                    # The privateKey field contains the secret
                    self.api_key = key_data['name']
                    self.api_secret = key_data['privateKey']
                    
                    logger.info("Coinbase CDP API credentials loaded successfully")
                else:
                    logger.warning(f"Invalid API key format in {self.api_key_file}")
                    logger.warning("Using public endpoints only")
                    self.api_key = None
                    self.api_secret = None
            else:
                logger.warning(f"API key file {self.api_key_file} not found")
                logger.warning("Using public endpoints only")
                self.api_key = None
                self.api_secret = None
                
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            logger.warning("Using public endpoints only")
            self.api_key = None
            self.api_secret = None
    
    def _rate_limit(self):
        """Implement rate limiting to respect Coinbase API limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_product_candles(self, product_id: str, start: str = None, end: str = None, 
                           granularity: str = 'FIFTEEN_MINUTE') -> List[Dict]:
        """
        Get candles for a specific product and time range using the official Coinbase SDK.
        
        Args:
            product_id: Product ID (e.g., 'XRP-USD')
            start: Start time (Unix timestamp or ISO string)
            end: End time (Unix timestamp or ISO string)
            granularity: Candle granularity (default: FIFTEEN_MINUTE)
            
        Returns:
            List of candle data
        """
        if not self.client:
            logger.error("No authenticated client available")
            return []
        
        self._rate_limit()
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                params = {
                    'product_id': product_id,
                    'granularity': granularity
                }
                
                if start:
                    params['start'] = start
                if end:
                    params['end'] = end
                    
                logger.info(f"Requesting candles with params: {params} (attempt {attempt + 1}/{max_retries})")
                candles_response = self.client.get_candles(**params)
     
                # Handle the response object
                if hasattr(candles_response, 'candles'):
                    return candles_response.candles
                elif hasattr(candles_response, '__dict__'):
                    return candles_response.__dict__.get('candles', [])
                else:
                    return candles_response
                    
            except Exception as e:
                logger.error(f"Error fetching candles (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached, giving up")
                    return []
        
        return []
    
    def get_accounts(self) -> List[Dict]:
        """
        Get account information (requires authentication).
        
        Returns:
            List of account information
        """
        if not self.client:
            logger.warning("No authenticated client available")
            return []
        
        try:
            accounts_response = self.client.get_accounts()
            if hasattr(accounts_response, 'accounts'):
                return accounts_response.accounts
            elif hasattr(accounts_response, '__dict__'):
                return accounts_response.__dict__.get('accounts', [])
            else:
                return list(accounts_response) if accounts_response else []
        except Exception as e:
            logger.error(f"Error fetching accounts: {e}")
            return []
    
    def get_account_balance(self, account_id: str) -> Dict:
        """
        Get balance for a specific account (requires authentication).
        
        Args:
            account_id: Account ID
            
        Returns:
            Account balance information
        """
        if not self.client:
            logger.warning("No authenticated client available")
            return {}
        
        try:
            account_response = self.client.get_account(account_id=account_id)
            if hasattr(account_response, '__dict__'):
                return account_response.__dict__
            else:
                return account_response
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return {}
    
    def get_products(self) -> List[Dict]:
        """
        Get list of available products (requires authentication).
        
        Returns:
            List of available products
        """
        if not self.client:
            logger.warning("No authenticated client available")
            return []
        
        try:
            products_response = self.client.get_products()
            if hasattr(products_response, 'products'):
                return products_response.products
            elif hasattr(products_response, '__dict__'):
                return products_response.__dict__.get('products', [])
            else:
                return list(products_response) if products_response else []
        except Exception as e:
            logger.error(f"Error fetching products: {e}")
            return []
    
    def check_product_exists(self, product_id: str) -> bool:
        """
        Check if a specific product exists and is available for trading.
        
        Args:
            product_id: Product ID to check
            
        Returns:
            True if product exists and is available
        """
        if not self.client:
            logger.warning("No authenticated client available")
            return False
        
        try:
            products = self.get_products()
            logger.debug(f"Checking {len(products)} products for {product_id}")
            
            for product in products:
                current_product_id = None
                status = None
                
                if hasattr(product, 'product_id'):
                    current_product_id = product.product_id
                    status = getattr(product, 'status', None)
                elif isinstance(product, dict):
                    current_product_id = product.get('product_id')
                    status = product.get('status', None)
                
                if current_product_id == product_id:
                    logger.debug(f"Found product {product_id} with status: {status}")
                    # If status is present, check if it's ONLINE, otherwise assume available
                    if status is not None:
                        return status in ['ONLINE', 'online', 'Online']
                    else:
                        return True  # Assume available if no status field
            
            logger.debug(f"Product {product_id} not found in products list")
            return False
        except Exception as e:
            logger.error(f"Error checking product existence: {e}")
            return False

    def pull_historical_data(self, product_id: str, start_date: str = '2020-01-01', 
                           end_date: str = None, granularity: str = 'FIFTEEN_MINUTE') -> pd.DataFrame:
        """
        Pull all historical data from start_date to end_date, handling the 350 bar limit.
        
        Args:
            product_id: Product ID (e.g., 'XRP-USD')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to current date)
            granularity: Candle granularity
            
        Returns:
            DataFrame with all historical data
        """
        if not self.client:
            logger.error("No authenticated client available. Please check your API credentials.")
            return pd.DataFrame()
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Convert dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Calculate time intervals based on granularity
        granularity_minutes = self._get_granularity_minutes(granularity)
        max_bars_per_request = 300  # Use 300 to be safe (Coinbase limit is 350)
        
        # Calculate time span for 300 bars
        max_time_span = timedelta(minutes=granularity_minutes * max_bars_per_request)
        
        all_candles = []
        current_start = start_dt
        
        logger.info(f"Starting data pull for {product_id} from {start_date} to {end_date}")
        logger.info(f"Granularity: {granularity} ({granularity_minutes} minutes)")
        logger.info(f"Max bars per request: {max_bars_per_request}")
        
        request_count = 0
        
        while current_start < end_dt:
            # Calculate end time for this batch
            current_end = min(current_start + max_time_span, end_dt)
            
            # Convert to Unix timestamps for the Coinbase SDK
            start_timestamp = int(current_start.timestamp())
            end_timestamp = int(current_end.timestamp())
            
            try:
                logger.info(f"Fetching candles for {product_id} from {current_start.strftime('%Y-%m-%dT%H:%M:%SZ')} to {current_end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
                candles_data = self.get_product_candles(
                    product_id=product_id,
                    start=start_timestamp,
                    end=end_timestamp,
                    granularity=granularity
                )
                
                if candles_data and len(candles_data) > 0:
                    all_candles.extend(candles_data)
                    logger.info(f"Fetched {len(candles_data)} candles from {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}")
                else:
                    logger.warning(f"No data returned for period {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}")
                
                request_count += 1
                
                # Progress update every 10 requests
                if request_count % 10 == 0:
                    progress = (current_start - start_dt) / (end_dt - start_dt) * 100
                    logger.info(f"Progress: {progress:.1f}% ({request_count} requests made)")
                
            except Exception as e:
                logger.error(f"Error fetching data for period {current_start.strftime('%Y-%m-%d %H:%M')} to {current_end.strftime('%Y-%m-%d %H:%M')}: {e}")
                # Continue with next period
                pass
            
            # Move to next batch
            current_start = current_end
        
        logger.info(f"Data pull completed. Total candles fetched: {len(all_candles)}")
        
        # Convert to DataFrame
        if all_candles:
            df = self._convert_candles_to_dataframe(all_candles)
            return df
        else:
            logger.warning("No data was fetched")
            return pd.DataFrame()
    
    def _get_granularity_minutes(self, granularity: str) -> int:
        """Convert granularity string to minutes."""
        granularity_map = {
            'ONE_MINUTE': 1,
            'FIVE_MINUTE': 5,
            'FIFTEEN_MINUTE': 15,
            'THIRTY_MINUTE': 30,
            'ONE_HOUR': 60,
            'TWO_HOUR': 120,
            'SIX_HOUR': 360,
            'ONE_DAY': 1440
        }
        return granularity_map.get(granularity, 15)
    
    def _convert_candles_to_dataframe(self, candles_data: List[Dict]) -> pd.DataFrame:
        """
        Convert Coinbase candle data to a pandas DataFrame.
        
        Args:
            candles_data: List of candle objects from Coinbase SDK
            
        Returns:
            DataFrame with columns: timestamp, Open, High, Low, Close, Volume
            where timestamp is formatted as 'YYYY-MM-DD HH:MM:SS'
        """
        if not candles_data:
            return pd.DataFrame()
        
        # Convert candle objects to a list of dictionaries
        processed_candles = []
        
        for candle in candles_data:
            # Handle different response formats from Coinbase SDK
            if hasattr(candle, 'start'):
                # Candle object with attributes
                processed_candle = {
                    'timestamp': candle.start,
                    'low': float(candle.low),
                    'high': float(candle.high),
                    'open': float(candle.open),
                    'close': float(candle.close),
                    'volume': float(candle.volume)
                }
            elif isinstance(candle, dict):
                # Dictionary format
                processed_candle = {
                    'timestamp': candle.get('start', candle.get('timestamp', 0)),
                    'low': float(candle.get('low', 0)),
                    'high': float(candle.get('high', 0)),
                    'open': float(candle.get('open', 0)),
                    'close': float(candle.get('close', 0)),
                    'volume': float(candle.get('volume', 0))
                }
            else:
                # Fallback: try to access as object attributes
                try:
                    processed_candle = {
                        'timestamp': getattr(candle, 'start', 0),
                        'low': float(getattr(candle, 'low', 0)),
                        'high': float(getattr(candle, 'high', 0)),
                        'open': float(getattr(candle, 'open', 0)),
                        'close': float(getattr(candle, 'close', 0)),
                        'volume': float(getattr(candle, 'volume', 0))
                    }
                except:
                    logger.warning(f"Could not process candle: {candle}")
                    continue
            
            processed_candles.append(processed_candle)
        
        if not processed_candles:
            logger.warning("No valid candles found in the data")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(processed_candles)
        
        # Convert timestamp to datetime (Coinbase SDK uses Unix timestamps)
        if df['timestamp'].dtype == 'object':
            # If timestamp is a string, try to parse as ISO or convert to int
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                # Try converting string to int then to datetime
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            # If timestamp is numeric, assume it's Unix timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Convert string values to float
        numeric_columns = ['low', 'high', 'open', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename columns to match standard format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'timestamp': 'timestamp'
        })
        
        # Format timestamp as 'YYYY-MM-DD HH:MM:SS' and make it the first column
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Select and reorder columns with timestamp first
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def save_data(self, df: pd.DataFrame, product_id: str, granularity: str = 'FIFTEEN_MINUTE'):
        """
        Save the data to a CSV file with technical indicators calculated.
        
        Args:
            df: DataFrame to save
            product_id: Product ID for filename
            granularity: Granularity for filename
        """
        if df.empty:
            logger.warning("No data to save")
            return
        
        # Convert timestamp back to datetime for technical indicator calculations
        df_with_datetime = df.copy()
        df_with_datetime['timestamp'] = pd.to_datetime(df_with_datetime['timestamp'])
        df_with_datetime = df_with_datetime.set_index('timestamp')
        
        # Rename columns to match what calculate_technical_indicators expects
        df_with_datetime = df_with_datetime.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        try:
            # Get granularity in minutes for the timeframe parameter
            granularity_minutes = self._get_granularity_minutes(granularity)
            df_with_indicators = calculate_technical_indicators(df_with_datetime, market_data=None, timeframe=granularity_minutes)
            
            # Reset index to get timestamp back as a column
            df_with_indicators = df_with_indicators.reset_index()
            
            # Format timestamp as 'YYYY-MM-DD HH:MM:SS' and make it the first column
            df_with_indicators['timestamp'] = df_with_indicators['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Reorder columns to put timestamp first
            cols = ['timestamp'] + [col for col in df_with_indicators.columns if col != 'timestamp']
            df_with_indicators = df_with_indicators[cols]
            
            logger.info(f"Technical indicators calculated successfully. New shape: {df_with_indicators.shape}")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            logger.warning("Saving data without technical indicators")
            df_with_indicators = df
        
        # Create filename
        granularity_clean = granularity.lower().replace('_', '')
        filename = f"DB/{product_id.replace('-', '_')}_{granularity_clean}_indicators.csv"
        
        # Save to CSV
        df_with_indicators.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
        logger.info(f"Data shape: {df_with_indicators.shape}")
        logger.info(f"Date range: {df_with_indicators['timestamp'].min()} to {df_with_indicators['timestamp'].max()}")
        
        return filename


def main():
    """Main function to demonstrate usage."""
    
    # Initialize the data puller (will automatically load API keys from file)
    puller = CoinbaseDataPuller()
    
    # Check if we have API credentials
    if puller.api_key and puller.api_secret:
        logger.info("API credentials loaded successfully")
        
        # Try to get account information
        try:
            accounts = puller.get_accounts()
            if accounts:
                logger.info(f"Found {len(accounts)} accounts")
                for account in accounts[:3]:  # Show first 3 accounts
                    if hasattr(account, 'currency'):
                        currency = account.currency
                        balance = getattr(account, 'available_balance', {})
                        if hasattr(balance, 'value'):
                            balance_value = balance.value
                        else:
                            balance_value = str(balance)
                    elif isinstance(account, dict):
                        currency = account.get('currency', 'Unknown')
                        balance_value = account.get('balance', '0')
                    else:
                        currency = 'Unknown'
                        balance_value = '0'
                    logger.info(f"Account: {currency} - Balance: {balance_value}")
            else:
                logger.info("No accounts found or access denied")
        except Exception as e:
            logger.warning(f"Could not fetch account information: {e}")
        
        # Check available products
        try:
            products = puller.get_products()
            if products:
                logger.info(f"Found {len(products)} available products")
                # Show some popular products
                popular_products = ['XRP-USD', 'BTC-USD', 'ETH-USD', 'ADA-USD']
                for product_id in popular_products:
                    exists = puller.check_product_exists(product_id)
                    status = "Available" if exists else "Not Available"
                    logger.info(f"Product {product_id}: {status}")
            else:
                logger.warning("Could not fetch product list")
        except Exception as e:
            logger.warning(f"Could not fetch product information: {e}")
    else:
        logger.error("No API credentials available - cannot proceed with data pulling")
        logger.error("Please check your settings/cdp_api_key.json file")
        return
    
    # Define parameters
    product_id = 'XRP-USD'  # Change this to your desired product
    
    # Check if the product exists before proceeding
    if not puller.check_product_exists(product_id):
        logger.error(f"Product {product_id} is not available for trading")
        logger.info("Available products include:")
        try:
            products = puller.get_products()
            for product in products[:10]:  # Show first 10 products
                if hasattr(product, 'product_id'):
                    logger.info(f"  {product.product_id}")
                elif isinstance(product, dict):
                    logger.info(f"  {product.get('product_id', 'Unknown')}")
        except:
            pass
        return
    
    start_date = '2017-05-18'
    end_date = datetime.now().strftime('%Y-%m-%d')
    granularity = 'FIFTEEN_MINUTE'
    
    logger.info("Starting Coinbase historical data pull...")
    logger.info(f"Product: {product_id}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Granularity: {granularity}")
    
    try:
        # Pull the data
        df = puller.pull_historical_data(
            product_id=product_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity
        )
        
        if not df.empty:
            # Save the data
            filename = puller.save_data(df, product_id, granularity)
            
            # Display summary
            print("\n" + "="*50)
            print("DATA PULL SUMMARY")
            print("="*50)
            print(f"Product: {product_id}")
            print(f"Total candles: {len(df)}")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Granularity: {granularity}")
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
