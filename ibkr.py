from ib_insync import *
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import time
from GatherData import calculate_technical_indicators

class IBKRConnection:
    def __init__(self, host='127.0.0.1', port=7496, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
    def connect(self):
        """Connect to TWS or IB Gateway"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            print("Successfully connected to IBKR")
            return True
        except Exception as e:
            print(f"Failed to connect to IBKR: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("Disconnected from IBKR")
            
    def get_historical_data(self, symbol, duration='7 D', bar_size='15 mins', num_bars=None):
        """Get historical data for a symbol"""
        if not self.connected:
            print("Not connected to IBKR")
            return None
            
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Calculate duration based on number of bars if specified
        if num_bars is not None:
            # Convert bar_size to minutes
            if 'min' in bar_size:
                minutes = int(bar_size.split()[0])
            elif 'hour' in bar_size:
                minutes = int(bar_size.split()[0]) * 60
            elif 'day' in bar_size:
                minutes = int(bar_size.split()[0]) * 60 * 24
            else:
                minutes = 1
                
            # Calculate total minutes needed
            total_minutes = num_bars * minutes
            # Convert to trading days (6.5 hours per day), adding buffer
            trading_days = (total_minutes / 390) * 1.2  # 20% buffer
            duration = f"{int(trading_days)} D"
            
            print(f"Calculated duration: {trading_days:.1f} trading days")
            
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        df = pd.DataFrame(bars)
        if not df.empty:
            # Convert date column to datetime first
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Capitalize first letter of each column name
            df.columns = [col.capitalize() for col in df.columns]
            
            # Trim to exact number of bars if specified
            if num_bars is not None:
                df = df.tail(num_bars)
        return df
        
    def place_order(self, symbol, quantity, action, order_type='MKT'):
        """Place an order"""
        if not self.connected:
            print("Not connected to IBKR")
            return None
            
        contract = Stock(symbol, 'SMART', 'USD')
        if order_type == 'MKT':
            order = MarketOrder(action, quantity)
        elif order_type == 'LMT':
            order = LimitOrder(action, quantity, self.ib.reqMktData(contract).last)
            
        trade = self.ib.placeOrder(contract, order)
        return trade
        
    def get_account_summary(self):
        """Get account summary"""
        if not self.connected:
            print("Not connected to IBKR")
            return None
            
        account = self.ib.managedAccounts()[0]
        self.ib.reqAccountSummary()
        time.sleep(1)  # Wait for data
        return self.ib.accountSummary(account)
        
    def get_positions(self):
        """Get current positions"""
        if not self.connected:
            print("Not connected to IBKR")
            return None
            
        return self.ib.positions()
        
    def get_market_correlation_data(self, num_bars=200, bar_size='15 mins'):
        """Get historical data for market correlation symbols (SPY, QQQ, DIA, VXX, UVXY)
        
        Args:
            num_bars (int): Number of bars to retrieve
            bar_size (str): Bar size (e.g., '15 mins', '1 hour', '1 day')
            
        Returns:
            dict: Dictionary of dataframes for each market symbol
        """
        if not self.connected:
            print("Not connected to IBKR")
            return None
            
        market_symbols = ['SPY', 'QQQ', 'DIA', 'VXX', 'UVXY']
        market_data = {}
        
        for symbol in market_symbols:
            print(f"\nFetching data for {symbol}...")
            df = self.get_historical_data(symbol, num_bars=num_bars, bar_size=bar_size)
            
            if df is not None and not df.empty:
                # Ensure index is datetime
                df.index = pd.to_datetime(df.index)
                
                # Fill missing values using backward fill first, then forward fill
                df = df.bfill().ffill()
                
                # If any NaN values remain, replace with zeros (should be rare)
                df.fillna(0, inplace=True)
                
                # Remove any duplicate indices
                df = df[~df.index.duplicated(keep='first')]
                
                market_data[symbol] = df
                print(f"Retrieved {len(df)} bars for {symbol}")
            else:
                print(f"Failed to retrieve data for {symbol}")
                
        return market_data
        
    def get_live_data_with_indicators(self, symbol, num_bars=200, bar_size='15 mins'):
        """Get live data with technical indicators for a symbol and its market correlations
        
        Args:
            symbol (str): Stock symbol
            num_bars (int): Number of bars to retrieve (will fetch 2x this number for calculations)
            bar_size (str): Bar size (e.g., '15 mins', '1 hour', '1 day')
            
        Returns:
            tuple: (stock_df, market_data) where stock_df has all indicators calculated
        """
        if not self.connected:
            print("Not connected to IBKR")
            return None, None
            
        # Fetch double the requested bars to ensure proper indicator calculation (since some indicators require previous bars, use these previous bars to calculate them then trim them away)
        fetch_bars = num_bars * 2
        print(f"\nFetching {fetch_bars} bars for {symbol} (will return {num_bars} bars after calculation)...")
        stock_df = self.get_historical_data(symbol, num_bars=fetch_bars, bar_size=bar_size)
        
        if stock_df is None or stock_df.empty:
            print(f"Failed to retrieve data for {symbol}")
            return None, None
            
        print(f"Retrieved {len(stock_df)} bars for {symbol}")
        
        # Get market correlation data with same number of bars
        print("\nFetching market correlation data...")
        market_data = self.get_market_correlation_data(num_bars=fetch_bars, bar_size=bar_size)
        
        if market_data is None:
            print("Failed to retrieve market correlation data")
            return stock_df, None
            
        # Calculate technical indicators
        print("\nCalculating technical indicators...")
        stock_df = calculate_technical_indicators(stock_df, market_data)
        
        # Trim to requested number of bars
        stock_df = stock_df.tail(num_bars)
        
        # Also trim market data to match
        for market_symbol in market_data:
            market_data[market_symbol] = market_data[market_symbol].tail(num_bars)
            
        return stock_df, market_data
        
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

# Example usage
if __name__ == "__main__":
    # Connect to TWS
    symbol = "TSLA"
    ib = IBKRConnection()
    if ib.connect():
        try:
            LOOKBACK_BARS = 200
            # Get first batch of live data with indicators to feed into the model
            model_window_df, market_data = ib.get_live_data_with_indicators(symbol, num_bars=LOOKBACK_BARS)
            
            if model_window_df is not None:
                print(f"\nStock Data Summary:")
                print(f"Number of bars: {len(model_window_df)}")
                print(f"Date range: {model_window_df.index.min()} to {model_window_df.index.max()}")
                print("\nFirst few rows with indicators:")
                model_window_df.to_csv("DB/ibkr_test.csv")
                print(model_window_df.head())
                

                    
        finally:
            ib.disconnect() 