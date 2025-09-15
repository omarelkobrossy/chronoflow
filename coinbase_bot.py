# Coinbase API integration using official Python SDK
# Uses CDP API key from settings/cdp_api_key.json

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os
import uuid
import joblib
import xgboost as xgb
import time
import threading
from json import dumps
from coinbase.rest import RESTClient

# Add the Quant directory to the path to import from GatherData.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'Quant'))
from GatherData import calculate_technical_indicators
from utils import preprocess_data, calculate_feature_importance

class CoinbaseAPIClient:
    def __init__(self, api_key=None, api_secret=None, symbol='XRP-USD', years_of_data=2):
        """
        Initialize Coinbase API client using official SDK
        
        Args:
            api_key: API key name from CDP JSON file
            api_secret: Private key from CDP JSON file
            symbol: Trading pair (default: 'XRP-USD')
            years_of_data: Number of years of historical data to load (default: 2)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.years_of_data = years_of_data
        self.client = None
        self.model = None
        self.current_features = None
        self.window_size = 25000
        self.maker_fee = 0.006
        self.taker_fee = 0.012

        # In-memory data storage
        self.historical_data = pd.DataFrame()  # Full historical dataset with indicators
        self.raw_data = pd.DataFrame()  # Raw OHLCV data without indicators
        
        # Incremental statistics for expanding standardization
        self.running_stats = {}  # Will store {feature: {'sum': x, 'sum_sq': x, 'count': n}}
        self.last_processed_timestamp = None
        self.retrain_interval = 15  # Number of 15-minute intervals between retrains (15 * 15 = 225 minutes)
        self.last_retrain_idx = 0
        self.last_retrain_time = None
        self.last_update_time = None
        self.trading_params = {
            'min_risk_percentage': 0.005050715939253347,
            'max_risk_percentage': 0.03742776224889559,
            'risk_scaling_factor': 2.9122956569535874,#1.7429109176364634,#1.6949727011806939,
            'risk_reward_ratio': 1.7553062836156037,#1.6037213842177254,
            'min_predicted_move': 0.004275510161827971,#0.005113433706915217,
            'partial_take_profit': 0.7796142341082877,
            'min_holding_period': 10,
            'max_holding_period': 11,
            'max_concurrent_trades': 9,
            'stop_loss_atr_multiplier': 3.085350201403653,
            'atr_predicted_weight': 0.36613925061300917,
        }
        self.trade_history = []
        self.capital = 0  # Will be updated from account balance
        self.order_monitor_thread = None
        self.stop_monitoring = False
        self.active_brackets = {}  # Track active bracket orders
        self.model_params = {
            "learning_rate": 0.1081270658051096,
            "n_estimators": 943,
            "max_depth": 5,
            "max_leaves": 123,
            "min_child_weight": 0.6266777639806653,
            "gamma": 0.1453711881569889,
            "subsample": 0.8905128494073881,
            "colsample_bytree": 0.4595148605371952,
            "colsample_bylevel": 0.5233925218009232,
            "reg_lambda": 1.4092734173101562,
            "reg_alpha": 1.4151336617040773,
            "max_bin": 563,
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'cpu'
        }
        
        if api_key and api_secret:
            self.client = RESTClient(api_key=api_key, api_secret=api_secret)
    
    def sync_with_account_orders(self):
        """Sync with actual open orders from Coinbase account and update capital"""
        try:
            if not self.client:
                print("No authenticated client available")
                return False
            
            # Update capital from account first
            print("Syncing with account...")
            if not self.update_capital_from_account():
                print("Warning: Failed to update capital from account")
            
            # Get all open orders
            orders_response = self.get_orders(status='OPEN')
            if not orders_response:
                print("No open orders found")
                return True
            
            # Convert to list if it's a response object
            if isinstance(orders_response, dict):
                orders = orders_response.get('orders', [])
            else:
                orders = list(orders_response)
            
            print(f"Found {len(orders)} open orders on account")
            
            # Filter orders for our symbol using direct attributes
            symbol_orders = []
            for order in orders:
                if hasattr(order, 'product_id'):
                    product_id = order.product_id
                elif isinstance(order, dict):
                    product_id = order.get('product_id', '')
                else:
                    product_id = ''
                    
                if product_id == self.symbol:
                    symbol_orders.append(order)
            
            if symbol_orders:
                print(f"Found {len(symbol_orders)} open orders for {self.symbol}")
                for order in symbol_orders:
                    # Get order details using direct attributes
                    if hasattr(order, 'order_id'):
                        order_id = order.order_id
                    elif isinstance(order, dict):
                        order_id = order.get('order_id', 'Unknown')
                    else:
                        order_id = 'Unknown'
                        
                    if hasattr(order, 'side'):
                        side = order.side
                    elif isinstance(order, dict):
                        side = order.get('side', 'Unknown')
                    else:
                        side = 'Unknown'
                        
                    if hasattr(order, 'total_size'):
                        size = order.total_size
                    elif isinstance(order, dict):
                        size = order.get('total_size', '0')
                    else:
                        size = '0'
                        
                    if hasattr(order, 'price'):
                        price = order.price
                    elif isinstance(order, dict):
                        price = order.get('price', '0')
                    else:
                        price = '0'
                        
                    if hasattr(order, 'order_type'):
                        order_type = order.order_type
                    elif isinstance(order, dict):
                        order_type = order.get('order_type', 'Unknown')
                    else:
                        order_type = 'Unknown'
                    
                    print(f"  - {side} {size} {self.symbol} @ {price} ({order_type}) - ID: {order_id}")
            else:
                print(f"No open orders for {self.symbol}")
            
            return True
            
        except Exception as e:
            print(f"Error syncing with account orders: {e}")
            return False
    
    def get_open_trades_count(self):
        """Get count of open trades from account"""
        try:
            if not self.client:
                return 0
            
            print("DEBUG: Getting orders...")
            orders_response = self.get_orders(status='OPEN')
            #print(f"DEBUG: Orders response: {orders_response}")
            
            if not orders_response:
                return 0
            
            # Convert to list if it's a response object
            if isinstance(orders_response, dict):
                orders = orders_response.get('orders', [])
            else:
                orders = list(orders_response)
            
            #print(f"DEBUG: Orders list: {orders}")
            
            # Count orders for our symbol using direct attributes
            symbol_orders = []
            for order in orders:
                if hasattr(order, 'product_id'):
                    product_id = order.product_id
                elif isinstance(order, dict):
                    product_id = order.get('product_id', '')
                else:
                    product_id = ''
                    
                if product_id == self.symbol:
                    symbol_orders.append(order)
                    
            count = len(symbol_orders)
            print(f"DEBUG: Symbol orders count: {count}")
            return count
            
        except Exception as e:
            print(f"Error getting open trades count: {e}")
            return 0
    
    def get_accounts(self):
        """Fetch account information"""
        try:
            if not self.client:
                print("No authenticated client available")
                return None
            
            accounts_response = self.client.get_accounts()
            # Convert to dictionary if it's a response object
            if hasattr(accounts_response, '__dict__'):
                return accounts_response.__dict__
            return accounts_response
        except Exception as e:
            print(f"Error fetching accounts: {e}")
            return None
    
    def get_usd_balance(self):
        """Get USD balance from account"""
        try:
            if not self.client:
                print("No authenticated client available")
                return None
            
            accounts_response = self.get_accounts()
            if not accounts_response:
                print("Failed to fetch accounts")
                return None
            
            print(f"Accounts response structure: {type(accounts_response)}")
            if hasattr(accounts_response, '__dict__'):
                print(f"Accounts response keys: {list(accounts_response.__dict__.keys())}")
            
            # Look for USD account
            accounts = accounts_response.get('accounts', [])
            #print(f"Found {len(accounts)} accounts")
            
            for i, account in enumerate(accounts):
                #print(f"Account {i}: {account}")
                
                # Handle account as object with attributes
                if hasattr(account, 'currency'):
                    currency = account.currency
                elif isinstance(account, dict):
                    currency = account.get('currency', 'Unknown')
                else:
                    currency = 'Unknown'
                    
                print(f"  Currency: {currency}")
                
                if currency == 'USD':
                    if hasattr(account, 'available_balance'):
                        available_balance = account.available_balance
                    elif isinstance(account, dict):
                        available_balance = account.get('available_balance', {})
                    else:
                        available_balance = {}
                        
                    print(f"  Available balance structure: {available_balance}")
                    
                    if hasattr(available_balance, 'value'):
                        balance_value = available_balance.value
                    elif isinstance(available_balance, dict):
                        balance_value = available_balance.get('value', '0')
                    else:
                        balance_value = str(available_balance)
                    
                    balance = float(balance_value)
                    print(f"USD Balance: ${balance:,.2f}")
                    return balance
            
            print("No USD account found")
            print("Available currencies:", [acc.currency if hasattr(acc, 'currency') else 'Unknown' for acc in accounts])
            return None
            
        except Exception as e:
            print(f"Error fetching USD balance: {e}")
            return None
    
    def update_capital_from_account(self):
        """Update capital from actual USD balance"""
        try:
            print("DEBUG: update_capital_from_account() called")
            usd_balance = self.get_usd_balance()
            print(f"DEBUG: get_usd_balance() returned: {usd_balance}")
            if usd_balance is not None:
                self.capital = usd_balance
                print(f"Updated capital from account: ${self.capital:,.2f}")
                return True
            else:
                print("Failed to update capital from account")
                return False
        except Exception as e:
            print(f"Error updating capital: {e}")
            return False
    
    def monitor_bracket_orders(self):
        """Monitor bracket orders and check their status"""
        while not self.stop_monitoring:
            try:
                # Check each active bracket
                for bracket_id, bracket_info in list(self.active_brackets.items()):
                    order_id = bracket_info['order_id']
                    created_time = bracket_info.get('created_time', 0)
                    
                    # Add grace period for new orders (30 seconds)
                    if time.time() - created_time < 30:
                        continue  # Skip checking orders that are less than 30 seconds old
                    
                    order_status = 'UNKNOWN'
                    
                    # Check if order is filled or cancelled
                    try:
                        order = self.client.get_order(order_id)
                        order_status = getattr(order, 'status', 'UNKNOWN')
                        
                        if order_status == 'FILLED':
                            print(f"Order {order_id} filled (either stop loss or take profit triggered)")
                            del self.active_brackets[bracket_id]
                            continue
                        elif order_status in ['CANCELLED', 'EXPIRED']:
                            print(f"Order {order_id} was cancelled/expired, removing from monitoring")
                            del self.active_brackets[bracket_id]
                            continue
                        else:
                            # Order is still active, check if we should cancel it due to max holding period
                            try:
                                order_details = self.client.get_order(order_id)
                                if hasattr(order_details, 'created_time'):
                                    created_time = datetime.fromisoformat(order_details.created_time.replace('Z', '+00:00'))
                                    holding_period = (datetime.now(timezone.utc) - created_time).total_seconds() / 60
                                    
                                    if holding_period >= self.trading_params['max_holding_period']:
                                        print(f"Cancelling old order {order_id} (held for {holding_period:.1f} minutes)")
                                        self.client.cancel_orders([order_id])
                                        del self.active_brackets[bracket_id]
                                        continue
                            except Exception as e:
                                print(f"Error checking order holding period {order_id}: {e}")
                                
                    except Exception as e:
                        if "NOT_FOUND" in str(e):
                            # Only log once per bracket to reduce noise
                            if bracket_id in getattr(self, '_logged_not_found', set()):
                                del self.active_brackets[bracket_id]
                                continue
                            else:
                                if not hasattr(self, '_logged_not_found'):
                                    self._logged_not_found = set()
                                self._logged_not_found.add(bracket_id)
                                print(f"Order {order_id} not found (likely already filled/cancelled), removing bracket {bracket_id}")
                                del self.active_brackets[bracket_id]
                                continue
                        else:
                            print(f"Error checking order {order_id}: {e}")
                
                # Sleep for 10 seconds before next check
                time.sleep(10)
                
            except Exception as e:
                print(f"Error in bracket order monitor: {e}")
                time.sleep(10)
    
    def start_order_monitor(self):
        """Start the order monitoring thread"""
        if self.order_monitor_thread is None or not self.order_monitor_thread.is_alive():
            self.stop_monitoring = False
            self.order_monitor_thread = threading.Thread(target=self.monitor_bracket_orders, daemon=True)
            self.order_monitor_thread.start()
            print("Order monitoring thread started")
    
    def stop_order_monitor(self):
        """Stop the order monitoring thread"""
        self.stop_monitoring = True
        if self.order_monitor_thread and self.order_monitor_thread.is_alive():
            self.order_monitor_thread.join(timeout=5)
            print("Order monitoring thread stopped")
    
    def get_orders(self, status='OPEN'):
        """Fetch current orders"""
        try:
            if not self.client:
                print("No authenticated client available")
                return None
            
            # Use the correct method name from the SDK
            orders_response = self.client.list_orders(order_status=status)
            
            # Convert to dictionary if it's a response object
            if hasattr(orders_response, '__dict__'):
                return orders_response.__dict__
            return orders_response
        except Exception as e:
            print(f"Error fetching orders: {e}")
            return None
    
    def initialize_model(self, df=None):
        """Initialize the model with historical data"""
        print("\nInitializing model...")
        
        # Use the full in-memory historical data (already has technical indicators)
        if df is None:
            if self.historical_data.empty:
                print("No historical data available for model initialization")
                return
            df = self.historical_data.copy()
        
        print(f"Using {len(df):,} bars from full historical dataset")
        
        # Data already has technical indicators calculated, just preprocess
        
        # Preprocess data using the utility function
        df_processed, feature_cols, target_cols = preprocess_data(df)
        
        # Calculate initial feature importance
        self.current_features = calculate_feature_importance(
            df_processed,
            feature_cols,
            target_cols,
            iterations=1,
            save_importance=False,
            visualize_importance=False
        )
        
        # Initialize running statistics from the in-memory dataset
        print("Loading running statistics from in-memory dataset...")
        if not self.initialize_running_stats_from_memory():
            print("Warning: Failed to initialize running stats from memory, using current data only")
            # Fallback to current data if memory loading fails
            self.running_stats = {}
            for col in self.current_features:
                if col in df_processed.columns:
                    values = df_processed[col].dropna()
                    self.running_stats[col] = {
                        'sum': float(values.sum()),
                        'sum_sq': float((values ** 2).sum()),
                        'count': len(values)
                    }
        
        self.model = xgb.XGBRegressor(**self.model_params)
        
        # Get preprocessed features and target using incremental standardization
        X_initial = []
        y_initial = []
        
        # Reset running stats for training (they're already initialized from CSV or fallback)
        temp_running_stats = {col: self.running_stats[col].copy() for col in self.current_features if col in self.running_stats}
        
        # Use only the most recent window_size bars for training (but with full context for indicators)
        # The buffer bars provided context for technical indicators, now we train on the recent window
        training_start_idx = max(0, len(df_processed) - self.window_size)
        training_df = df_processed.iloc[training_start_idx:].copy()
        
        print(f"Training on most recent {len(training_df)} bars (window_size: {self.window_size})")
        print(f"Full context: {len(df_processed)} bars (buffer provided proper indicator calculation)")
        
        # Process each row sequentially to build training data with incremental standardization
        # Start from beginning of full dataset to maintain proper running stats, but only save recent window for training
        for i in range(len(df_processed)):
            row = df_processed.iloc[i]
            row_features = {col: row[col] for col in self.current_features if col in df_processed.columns}
            
            # Get standardized features using current running stats
            if i > 0:  # Skip first row as we need previous data for standardization
                standardized_row = self.get_expanding_standardization_with_stats(row_features, temp_running_stats)
                
                # Only add to training data if we're in the training window
                if i >= training_start_idx:
                    X_initial.append([standardized_row.get(col, 0.0) for col in self.current_features])
                    y_initial.append(row[target_cols[0]])  # Assuming single target
            
            # Update temp running stats with current row (for all rows to maintain consistency)
            for col, value in row_features.items():
                if col in temp_running_stats and not np.isnan(value):
                    temp_running_stats[col]['sum'] += value
                    temp_running_stats[col]['sum_sq'] += value ** 2
                    temp_running_stats[col]['count'] += 1
        
        if X_initial:
            X_initial = np.array(X_initial)
            y_initial = np.array(y_initial)
            
            # Train model
            self.model.fit(X_initial, y_initial)
            
            # Log training sanity check
            self.sanity_log(self.model, X_initial[-1:], tag="INIT_TRAIN")
        print("Model initialized successfully")
        
    def update_model(self, df=None):
        """Update the model with new data"""
        # Check if it's time to retrain
        if not self.should_retrain_model():
            return
            
        retrain_minutes = self.retrain_interval * 15
        print(f"\nRetraining model at {datetime.now().strftime('%H:%M:%S')} (every {retrain_minutes} minutes)...")
        
        # Use the full in-memory historical data (already has technical indicators)
        if df is None:
            if self.historical_data.empty:
                print("No historical data available for model retraining")
                return
            df = self.historical_data.copy()
        
        if len(df) < self.window_size:
            print(f"Insufficient data for model retraining: {len(df)} < {self.window_size}")
            return
            
        print(f"Using {len(df):,} bars from full historical dataset for retraining")
        
        # Data already has technical indicators calculated, just preprocess
        
        # Preprocess data using the utility function
        df_processed, feature_cols, target_cols = preprocess_data(df)
        
        # Recalculate feature importance if needed
        if len(df) % self.window_size == 0:
            print("Recalculating feature importance...")
            self.current_features = calculate_feature_importance(
                df_processed,
                feature_cols,
                target_cols,
                iterations=1,
                save_importance=False,
                visualize_importance=False
            )
        
        # Get preprocessed features and target using incremental standardization
        X = []
        y = []
        
        # Use current running stats for retraining
        temp_running_stats = {col: self.running_stats[col].copy() for col in self.current_features if col in self.running_stats}
        
        # Use only the most recent window_size bars for retraining (but with full context for indicators)
        # The buffer bars provided context for technical indicators, now we retrain on the recent window
        training_start_idx = max(0, len(df_processed) - self.window_size)
        training_df = df_processed.iloc[training_start_idx:].copy()
        
        print(f"Retraining on most recent {len(training_df)} bars (window_size: {self.window_size})")
        print(f"Full context: {len(df_processed)} bars (buffer provided proper indicator calculation)")
        
        # Process each row sequentially to build training data with incremental standardization
        # Start from beginning of full dataset to maintain proper running stats, but only save recent window for training
        for i in range(len(df_processed)):
            row = df_processed.iloc[i]
            row_features = {col: row[col] for col in self.current_features if col in df_processed.columns}
            
            # Get standardized features using current running stats
            if i > 0:  # Skip first row as we need previous data for standardization
                standardized_row = self.get_expanding_standardization_with_stats(row_features, temp_running_stats)
                
                # Only add to training data if we're in the training window
                if i >= training_start_idx:
                    X.append([standardized_row.get(col, 0.0) for col in self.current_features])
                    y.append(row[target_cols[0]])  # Assuming single target
            
            # Update temp running stats with current row (for all rows to maintain consistency)
            for col, value in row_features.items():
                if col in temp_running_stats and not np.isnan(value):
                    temp_running_stats[col]['sum'] += value
                    temp_running_stats[col]['sum_sq'] += value ** 2
                    temp_running_stats[col]['count'] += 1
        
        if X:
            X = np.array(X)
            y = np.array(y)
            
            # Update model
            self.model.fit(X, y)
            
            # Log retraining sanity check
            self.sanity_log(self.model, X[-1:], tag="RETRAIN")
        print("Model updated successfully")
        
    def predict_next_bar(self, df=None):
        """Predict the next bar's price change using incremental standardization"""
        if self.model is None:
            return None
            
        # Use the full in-memory historical data (already has technical indicators)
        if df is None:
            if self.historical_data.empty:
                print("No historical data available for prediction")
                return None
            df = self.historical_data.copy()
        
        if len(df) < self.window_size:
            print(f"Insufficient data for prediction: {len(df)} < {self.window_size}")
            return None
            
        # Data already has technical indicators calculated, just preprocess
        
        # Preprocess data using the utility function
        df_processed, feature_cols, target_cols = preprocess_data(df)
        
        # Get the latest row features
        latest_row = df_processed.iloc[-1]
        new_features = {col: latest_row[col] for col in self.current_features if col in df_processed.columns}
        
        # Get standardized features using running statistics
        if self.running_stats:
            # Use incremental standardization
            standardized_features = self.get_expanding_standardization(new_features)
            
            # Update running stats with the new data point
            self.update_running_stats(new_features)
            
            # Create feature array for prediction
            X_pred = np.array([[standardized_features.get(col, 0.0) for col in self.current_features]])
        else:
            # Fallback to traditional expanding standardization if running stats not available
            print("Warning: Using fallback standardization method")
            X = df_processed[self.current_features].copy()
            for col in self.current_features:
                mean = X[col].expanding(min_periods=1).mean().shift(1)
                std = X[col].expanding(min_periods=1).std().shift(1)
                X[col] = (X[col] - mean) / (std + 1e-8)
            
            X_pred = X.iloc[[-1]].values
        
        # Sanity check logging before prediction
        current_timestamp = df['Date'].iloc[-1] if 'Date' in df.columns else datetime.now()
        self.sanity_log(self.model, X_pred, tag=str(current_timestamp))
        
        # Check leaf diversity for recent predictions batch
        self.log_batch_leaf_diversity(df_processed, feature_cols=self.current_features)
        
        # Make prediction
        prediction = self.model.predict(X_pred)[0]
        return prediction
    
    def should_retrain_model(self):
        """Check if it's time to retrain the model based on retrain_interval * 15 minutes"""
        now = datetime.now()
        
        # If this is the first time, retrain immediately
        if self.last_retrain_time is None:
            self.last_retrain_time = now
            return True
        
        # Calculate minutes since last retrain
        minutes_since_last = (now - self.last_retrain_time).total_seconds() / 60
        
        # Retrain every retrain_interval * 15 minutes
        retrain_minutes = self.retrain_interval * 15
        if minutes_since_last >= retrain_minutes:
            self.last_retrain_time = now
            return True
        
        return False
    
    def should_update_data(self):
        """Check if it's time to pull new data and make predictions (every 15 minutes on the clock)"""
        now = datetime.now()
        
        # If this is the first time, update immediately
        if self.last_update_time is None:
            self.last_update_time = now
            return True
        
        # Check if current minute is divisible by 15 (0, 15, 30, 45)
        current_minute = now.minute
        if current_minute % 15 == 0:
            # Only update if we haven't already updated at this exact minute
            if self.last_update_time.minute != current_minute or self.last_update_time.hour != now.hour:
                self.last_update_time = now
                return True
        
        return False
    
    def execute_trade(self, prediction, current_price):
        """Execute a trade based on prediction"""
        # Check actual open trades from account
        open_trades_count = self.get_open_trades_count()
        if open_trades_count >= self.trading_params['max_concurrent_trades']:
            print(f"Maximum concurrent trades reached ({open_trades_count}/{self.trading_params['max_concurrent_trades']})")
            return
        print(f"Prediction: {prediction}, Min Predicted Move: {-self.trading_params['min_predicted_move']}")
        if prediction < -self.trading_params['min_predicted_move']:
            entry_price = current_price
            
            predicted_move = abs(prediction)*3
            risk_percentage = min(
                self.trading_params['min_risk_percentage'] * (1 + (predicted_move / self.trading_params['min_predicted_move']) * self.trading_params['risk_scaling_factor']),
                self.trading_params['max_risk_percentage']
            )
            
            risk_amount = self.capital * risk_percentage
            
            # Calculate stop loss and take profit based on risk_reward_ratio only
            # # Use a fixed percentage for stop loss, then calculate take profit based on risk_reward_ratio
            # stop_loss_percentage = 0.01  # 1% stop loss from entry price
            # stop_loss = entry_price * (1 - stop_loss_percentage)
            # take_profit = entry_price * (1 + (stop_loss_percentage * self.trading_params['risk_reward_ratio']))

            # Calculate stop loss and take profit using hybrid ATR and predicted move approach
            atr_value = self.historical_data['ATR'].iloc[-1]# if 'ATR' in self.historical_data else entry_price * 0.01  # Fallback to 1% if ATR not available
            
            # Calculate stop loss distance using hybrid approach
            atr_stop_distance = atr_value * stop_loss_atr_multiplier
            predicted_stop_distance = entry_price * predicted_move  # Convert predicted move to price distance
            
            # Weighted combination of ATR and predicted move
            stop_loss_distance = (atr_predicted_weight * atr_stop_distance + 
                                (1 - atr_predicted_weight) * predicted_stop_distance)
            # Calculate fee compensation factors
            # For stop loss: we need to account for the fact that we already paid the maker fee
            # For take profit: we need to account for both maker fee (already paid) and taker fee (will be paid)
            maker_fee_factor = 1 + self.maker_fee  # Factor to account for maker fee already paid
            taker_fee_factor = 1 - self.taker_fee  # Factor to account for taker fee on exit
            
            # Adjust stop loss to compensate for maker fee (we already paid it, so we need less distance)
            stop_loss = entry_price - (stop_loss_distance / maker_fee_factor)
            
            # Adjust take profit to compensate for both maker and taker fees
            # We need to reach a higher price to achieve the desired net profit
            target_net_profit = stop_loss_distance * risk_reward_ratio
            # Calculate the gross price needed to achieve target net profit after fees
            take_profit = entry_price + (target_net_profit / (maker_fee_factor * taker_fee_factor))

            risk_per_share = entry_price - stop_loss
            
            if risk_per_share <= 0 or np.isnan(risk_per_share):
                return
                
            size = risk_amount / risk_per_share
            size = min(size, self.capital / entry_price)
            size = np.floor(size)
            
            if size <= 0:
                return
                
            # Place the order using Coinbase API with automatic stop loss and take profit
            try:
                # Generate unique order ID for buy order (bracket order will use the same ID)
                buy_order_id = str(uuid.uuid4())
                
                # Calculate expected XRP amount for the bracket order
                rounded_quote_size = round(risk_amount, 4)
                calculated_xrp = rounded_quote_size / current_price
                print(f"Expected XRP amount: {calculated_xrp:.4f} XRP")
                
                # Place market buy order with attached bracket order configuration
                # Round prices to 4 decimal places for Coinbase precision requirements
                rounded_stop_loss = round(stop_loss, 4)
                rounded_take_profit = round(take_profit, 4)
                
                print(f"Placing market buy order with attached bracket configuration:")
                print(f"  - Stop trigger price: ${rounded_stop_loss:.4f}")
                print(f"  - Take profit price: ${rounded_take_profit:.4f}")
                
                # Create the order with attached bracket configuration
                order_response = self.client.create_order(
                    client_order_id=buy_order_id,
                    product_id=self.symbol,
                    side='BUY',
                    order_configuration={
                        'market_market_ioc': {
                            'quote_size': str(rounded_quote_size)
                        }
                    },
                    attached_order_configuration={
                        'trigger_bracket_gtc': {
                            'limit_price': str(rounded_take_profit),
                            'stop_trigger_price': str(rounded_stop_loss)
                        }
                    }
                )
                print(order_response)
                
                # Add order to monitoring system (with delay to allow orders to propagate)
                bracket_id = f"bracket_{buy_order_id}"
                self.active_brackets[bracket_id] = {
                    'order_id': buy_order_id,  # The main order ID
                    'entry_price': entry_price,
                    'xrp_amount': calculated_xrp,
                    'stop_trigger_price': rounded_stop_loss,
                    'take_profit_price': rounded_take_profit,
                    'created_time': time.time()  # Track when bracket was created
                }
                
                # Store trade info for logging (but don't use self.open_trades since we use persistent orders)
                trade_info = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'size': size,
                    'entry_time': datetime.now(),
                    'order_response': order_response,
                    'order_id': buy_order_id
                }
                
                print(f"Opened new trade with attached bracket configuration:")
                print(f"  Entry: ${entry_price:.4f}")
                print(f"  Stop Trigger: ${rounded_stop_loss:.4f}")
                print(f"  Take Profit: ${rounded_take_profit:.4f}")
                print(f"  Size: {calculated_xrp:.2f}")
                print(f"  Order ID: {buy_order_id} (added to monitoring)")
                
                # Update capital from account after placing trade
                time.sleep(2)  # Wait for orders to process
                self.update_capital_from_account()
                
            except Exception as e:
                print(f"Error executing trade: {e}")
                # Cancel any orders that might have been placed
                try:
                    if 'order_response' in locals():
                        self.client.cancel_orders([buy_order_id])
                except:
                    pass
    
    def update_trades(self, current_price):
        """Update open trades and check for exits"""
        try:
            # Get current open orders from account
            orders_response = self.get_orders(status='OPEN')
            if not orders_response:
                return
            
            # Convert to list if it's a response object
            if isinstance(orders_response, dict):
                orders = orders_response.get('orders', [])
            else:
                orders = list(orders_response)
            
            # Filter orders for our symbol using direct attributes
            symbol_orders = []
            for order in orders:
                if hasattr(order, 'product_id'):
                    product_id = order.product_id
                elif isinstance(order, dict):
                    product_id = order.get('product_id', '')
                else:
                    product_id = ''
                    
                if product_id == self.symbol:
                    symbol_orders.append(order)
            
            if not symbol_orders:
                return
            
            print(f"Monitoring {len(symbol_orders)} open orders for {self.symbol}")
            
            # Check each order for potential manual exits
            for order in symbol_orders:
                # Get order details using direct attributes
                if hasattr(order, 'order_id'):
                    order_id = order.order_id
                elif isinstance(order, dict):
                    order_id = order.get('order_id')
                else:
                    order_id = None
                    
                if hasattr(order, 'order_type'):
                    order_type = order.order_type
                elif isinstance(order, dict):
                    order_type = order.get('order_type', 'Unknown')
                else:
                    order_type = 'Unknown'
                    
                if hasattr(order, 'side'):
                    side = order.side
                elif isinstance(order, dict):
                    side = order.get('side', 'Unknown')
                else:
                    side = 'Unknown'
                
                # Only monitor sell orders (stop loss and take profit)
                if side == 'SELL':
                    # Check if this is a stop loss or take profit order
                    # We can identify this by the order type or price
                    if hasattr(order, 'price'):
                        order_price = float(order.price)
                    elif isinstance(order, dict):
                        order_price = float(order.get('price', 0))
                    else:
                        order_price = 0.0
                    
                    # If it's a stop order, check if it should be cancelled due to time
                    if order_type == 'STOP_LIMIT':
                        # This is likely a stop loss order
                        # Check if we should cancel it due to max holding period
                        # Since we don't have entry time, we'll use a simpler approach
                        # Cancel stop orders that have been open too long
                        try:
                            # Get order details to check creation time
                            order_details = self.client.get_order(order_id)
                            if hasattr(order_details, 'created_time'):
                                created_time = datetime.fromisoformat(order_details.created_time.replace('Z', '+00:00'))
                                holding_period = (datetime.now(timezone.utc) - created_time).total_seconds() / 60
                                
                                if holding_period >= self.trading_params['max_holding_period']:
                                    print(f"Cancelling old stop loss order {order_id} (held for {holding_period:.1f} minutes)")
                                    self.client.cancel_orders([order_id])
                        except Exception as e:
                            print(f"Error checking order {order_id}: {e}")
                            
        except Exception as e:
            print(f"Error updating trades: {e}")
    
    def close_trade(self, order_id, current_price, reason):
        """Close a trade by cancelling the order and placing a market sell"""
        try:
            # Cancel the existing order
            self.client.cancel_orders([order_id])
            
            # Get order details to determine size
            order_details = self.client.get_order(order_id)
            base_size = getattr(order_details, 'base_size', getattr(order_details, 'total_size', '0'))
            
            # Place market sell order
            sell_order = self.client.market_order_sell(
                client_order_id=str(uuid.uuid4()),
                product_id=self.symbol,
                base_size=base_size
            )
            
            print(f"Closed trade: Cancelled order {order_id}, placed market sell for {base_size}")
            
            # Update capital from account after trade closure
            time.sleep(2)  # Wait for order to process
            self.update_capital_from_account()
            
        except Exception as e:
            print(f"Error closing trade {order_id}: {e}")
    
    def load_historical_data_from_api(self):
        """
        Load X years of historical data from API into memory.
        Maintains the same DataFrame structure as CSV loading for compatibility.
        """
        try:
            print(f"\n=== Loading {self.years_of_data} years of historical data from API ===")
            
            # Calculate total bars needed (15-minute intervals)
            # 1 year ≈ 365 days × 24 hours × 4 intervals = 35,040 bars
            bars_per_year = 365 * 24 * 4
            total_bars_needed = int(self.years_of_data * bars_per_year)
            
            print(f"Target: {total_bars_needed:,} bars ({self.years_of_data} years of 15-minute data)")
            
            # Coinbase API limit is 350 bars per request
            max_bars_per_request = 350  # Use 350 to be safe
            requests_needed = (total_bars_needed // max_bars_per_request) + 1
            
            print(f"Will need approximately {requests_needed} API requests")
            
            if requests_needed > 200:  # Safety limit
                print(f"⚠️  Too many requests needed ({requests_needed}). Limiting to 2 years maximum.")
                total_bars_needed = 2 * bars_per_year
                requests_needed = (total_bars_needed // max_bars_per_request) + 1
            
            # Collect all data
            all_candles_data = []
            end_time = datetime.now(timezone.utc)
            
            print(f"Starting data collection from {end_time.strftime('%Y-%m-%d %H:%M:%S')} backwards...")
            
            for i in range(min(requests_needed, 200)):  # Cap at 200 requests
                # Calculate time range for this request
                minutes_per_request = max_bars_per_request * 15
                current_start = end_time - timedelta(minutes=minutes_per_request)
                
                print(f"  Request {i+1:3d}/{min(requests_needed, 200)}: {current_start.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
                
                try:
                    candles = self.get_product_candles(
                        product_id=self.symbol,
                        start=int(current_start.timestamp()),
                        end=int(end_time.timestamp()),
                        granularity='FIFTEEN_MINUTE'
                    )
                    
                    if candles and len(candles) > 0:
                        all_candles_data.extend(candles)
                        print(f"    ✅ Got {len(candles)} bars (Total: {len(all_candles_data):,})")
                    else:
                        print(f"    ❌ No data returned")
                        
                    # Update end_time for next iteration
                    end_time = current_start
                    
                    # Rate limiting
                    time.sleep(0.2)  # 200ms between requests
                    
                    # Check if we have enough data
                    if len(all_candles_data) >= total_bars_needed:
                        print(f"    🎯 Reached target of {total_bars_needed:,} bars")
                        break
                        
                except Exception as e:
                    print(f"    ❌ Error in request {i+1}: {e}")
                    continue
            
            # Convert to DataFrame using the same method as existing code
            if all_candles_data:
                print(f"\n📊 Converting {len(all_candles_data):,} candles to DataFrame...")
                
                # Use the existing conversion function
                self.raw_data = convert_coinbase_candles_to_dataframe(all_candles_data)
                
                if not self.raw_data.empty:
                    # CRITICAL: Maintain exact same structure as CSV loading
                    # Convert DatetimeIndex to Date column (same as load_data_with_historical_buffer)
                    self.raw_data = self.raw_data.reset_index()
                    self.raw_data = self.raw_data.rename(columns={'index': 'Date'})
                    
                    # Sort by date to ensure proper chronological order
                    self.raw_data = self.raw_data.sort_values('Date').reset_index(drop=True)
                    
                    print(f"✅ Successfully loaded {len(self.raw_data):,} bars of raw data")
                    print(f"   Date range: {self.raw_data['Date'].min()} to {self.raw_data['Date'].max()}")
                    print(f"   Columns: {list(self.raw_data.columns)}")
                    
                    # Calculate technical indicators for the full dataset
                    print(f"🔧 Calculating technical indicators...")
                    
                    # Prepare data for technical indicators (expects Date as index)
                    data_for_indicators = self.raw_data.copy()
                    if 'Date' in data_for_indicators.columns:
                        data_for_indicators = data_for_indicators.set_index('Date')
                    
                    self.historical_data = calculate_technical_indicators(data_for_indicators, None)
                    
                    # Clean up Date ambiguity: ensure Date is ONLY a column, not index
                    # calculate_technical_indicators creates both Date index AND Date column, fix this
                    has_date_column = 'Date' in self.historical_data.columns
                    has_date_index = hasattr(self.historical_data.index, 'name') and self.historical_data.index.name == 'Date'
                    
                    if has_date_column and has_date_index:
                        # We have both - drop the Date column and reset index to make it a column
                        self.historical_data = self.historical_data.drop(columns=['Date']).reset_index()
                    elif has_date_index:
                        # Only index is Date, reset to column
                        self.historical_data = self.historical_data.reset_index()
                    elif not has_date_column:
                        # No Date column, reset index to create it
                        self.historical_data = self.historical_data.reset_index()
                        if 'index' in self.historical_data.columns:
                            self.historical_data = self.historical_data.rename(columns={'index': 'Date'})
                    
                    print(f"✅ Historical data with indicators ready: {len(self.historical_data):,} bars")
                    print(f"   Total columns: {len(self.historical_data.columns)}")
                    
                    return True
                else:
                    print("❌ Failed to convert candle data to DataFrame")
                    return False
            else:
                print("❌ No candle data collected")
                return False
                
        except Exception as e:
            print(f"❌ Error loading historical data from API: {e}")
            return False
    

    

    
    def initialize_running_stats_from_memory(self):
        """Initialize running statistics from the in-memory historical dataset"""
        try:
            print("Initializing running statistics from in-memory dataset...")
            
            if self.historical_data.empty:
                print("❌ No historical data in memory. Load data first.")
                return False
            
            # Use the existing historical data with indicators
            df = self.historical_data.copy()
            
            # Preprocess the data (it already has indicators calculated)
            df_processed, feature_cols, target_cols = preprocess_data(df)
            
            # Initialize running stats for each feature
            self.running_stats = {}
            
            for col in self.current_features:
                if col in df_processed.columns:
                    values = df_processed[col].dropna()
                    self.running_stats[col] = {
                        'sum': float(values.sum()),
                        'sum_sq': float((values ** 2).sum()),
                        'count': len(values)
                    }
            
            # Track the last processed timestamp
            if 'Date' in self.historical_data.columns:
                self.last_processed_timestamp = self.historical_data['Date'].iloc[-1]
            else:
                print("❌ No Date column found in historical data")
                self.last_processed_timestamp = None
            
            print(f"✅ Initialized running stats for {len(self.running_stats)} features")
            print(f"   Last processed timestamp: {self.last_processed_timestamp}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing running stats from memory: {e}")
            return False
    
    def initialize_running_stats_from_csv(self, csv_file_path='DB/XRP_USD_fifteenminute_historical.csv'):
        """Initialize running statistics from the full historical dataset (LEGACY - kept for compatibility)"""
        try:
            print("Initializing running statistics from full dataset...")
            
            # Load full dataset
            df = pd.read_csv(csv_file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            # Calculate technical indicators and preprocess
            df = calculate_technical_indicators(df, None)
            df_processed, feature_cols, target_cols = preprocess_data(df)
            
            # Initialize running stats for each feature
            self.running_stats = {}
            
            for col in self.current_features:
                if col in df_processed.columns:
                    values = df_processed[col].dropna()
                    self.running_stats[col] = {
                        'sum': float(values.sum()),
                        'sum_sq': float((values ** 2).sum()),
                        'count': len(values)
                    }
            
            # Track the last processed timestamp
            self.last_processed_timestamp = df_processed.index[-1]
            
            print(f"Initialized running stats for {len(self.running_stats)} features")
            print(f"Last processed timestamp: {self.last_processed_timestamp}")
            
            return True
            
        except Exception as e:
            print(f"Error initializing running stats: {e}")
            return False
    
    def update_running_stats(self, new_features):
        """Update running statistics with new data point"""
        for col, value in new_features.items():
            if col in self.running_stats and not np.isnan(value):
                self.running_stats[col]['sum'] += value
                self.running_stats[col]['sum_sq'] += value ** 2
                self.running_stats[col]['count'] += 1
    
    def get_expanding_standardization(self, new_features):
        """Get standardized values using running statistics"""
        standardized = {}
        
        for col, value in new_features.items():
            if col in self.running_stats:
                stats = self.running_stats[col]
                count = stats['count']
                
                if count > 1:
                    # Calculate expanding mean and std
                    mean = stats['sum'] / count
                    variance = (stats['sum_sq'] / count) - (mean ** 2)
                    std = np.sqrt(max(variance, 1e-8))  # Avoid division by zero
                    
                    # Standardize the new value
                    standardized[col] = (value - mean) / (std + 1e-8)
                else:
                    standardized[col] = 0.0  # First data point
            else:
                standardized[col] = 0.0  # Feature not in running stats
        
        return standardized
    
    def get_expanding_standardization_with_stats(self, new_features, temp_stats):
        """Get standardized values using provided running statistics (for training)"""
        standardized = {}
        
        for col, value in new_features.items():
            if col in temp_stats:
                stats = temp_stats[col]
                count = stats['count']
                
                if count > 1:
                    # Calculate expanding mean and std
                    mean = stats['sum'] / count
                    variance = (stats['sum_sq'] / count) - (mean ** 2)
                    std = np.sqrt(max(variance, 1e-8))  # Avoid division by zero
                    
                    # Standardize the new value
                    standardized[col] = (value - mean) / (std + 1e-8)
                else:
                    standardized[col] = 0.0  # First data point
            else:
                standardized[col] = 0.0  # Feature not in running stats
        
        return standardized
    
    def sanity_log(self, model, X_row, tag, log_file_path='DB/sanity_log.txt'):
        """
        Log model and prediction sanity checks to a file
        
        Args:
            model: XGBoost model
            X_row: Input features (numpy array or pandas row)
            tag: Identifier for this log entry
            log_file_path: Path to log file
        """
        try:
            import os
            from datetime import datetime
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            # Get model info
            booster = model.get_booster()
            num_trees = booster.num_boosted_rounds()
            
            # Handle different input types
            if hasattr(X_row, 'values'):
                # Pandas DataFrame/Series
                values = X_row.values
                X_for_apply = X_row.values
            elif isinstance(X_row, np.ndarray):
                # Numpy array
                values = X_row
                X_for_apply = X_row
            else:
                # List or other
                values = np.array(X_row)
                X_for_apply = values
            
            # Flatten values for stats but keep original shape for model.apply
            if values.ndim > 1:
                values = values.flatten()
            
            # Calculate basic stats
            max_abs = np.nanmax(np.abs(values)) if len(values) > 0 else 0.0
            nan_percent = np.mean(~np.isfinite(values)) * 100 if len(values) > 0 else 0.0
            
            # Calculate leaf indices for diversity check
            leaf_diversity_info = ""
            try:
                # Ensure X_for_apply has correct shape for model.apply
                if X_for_apply.ndim == 1:
                    X_for_apply = X_for_apply.reshape(1, -1)
                
                # Get leaf indices for current sample(s)
                leaf_idx = model.apply(X_for_apply)  # shape: [n_samples, n_trees]
                
                # If we have multiple samples, check diversity
                if leaf_idx.shape[0] > 1:
                    uniq = len(np.unique(leaf_idx, axis=0))
                    total = leaf_idx.shape[0]
                    leaf_diversity_info = f"  leaf_uniq={uniq}/{total}"
                else:
                    # For single sample, just show the first few leaf indices
                    leaf_sample = leaf_idx[0][:min(5, len(leaf_idx[0]))]
                    leaf_diversity_info = f"  leaf_sample=[{','.join(map(str, leaf_sample))}...]"
                    
            except Exception as e:
                leaf_diversity_info = f"  leaf_err={str(e)[:20]}"
            
            # Create log entry
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = (f"[{timestamp}] [{tag}] trees={num_trees}  "
                        f"max|x|={max_abs:.4g}  "
                        f"nan%={nan_percent:.1f}%  "
                        f"features={len(values)}{leaf_diversity_info}\n")
            
            # Append to log file
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            # Also print to console
            print(f"[SANITY] [{tag}] trees={num_trees}  "
                  f"max|x|={max_abs:.4g}  "
                  f"nan%={nan_percent:.1f}%{leaf_diversity_info}")
            
        except Exception as e:
            print(f"Error in sanity_log: {e}")
    
    def log_batch_leaf_diversity(self, df_processed, feature_cols, Kb=16):
        """
        Log leaf index diversity for the last Kb rows to detect model stagnation
        
        Args:
            df_processed: Preprocessed dataframe with features
            feature_cols: List of feature column names
            Kb: Number of recent rows to check (default 16)
        """
        try:
            if len(df_processed) < Kb:
                Kb = len(df_processed)
            
            if Kb <= 1:
                return
                
            # Get last Kb rows
            Xb_df = df_processed[feature_cols].tail(Kb)
            
            # Apply incremental standardization to each row
            X_batch = []
            temp_running_stats = {col: self.running_stats[col].copy() for col in feature_cols if col in self.running_stats}
            
            for i in range(len(Xb_df)):
                row = Xb_df.iloc[i]
                row_features = {col: row[col] for col in feature_cols if col in Xb_df.columns}
                
                # Get standardized features
                if i > 0:  # Skip first row for standardization
                    standardized_row = self.get_expanding_standardization_with_stats(row_features, temp_running_stats)
                    X_batch.append([standardized_row.get(col, 0.0) for col in feature_cols])
                
                # Update temp running stats
                for col, value in row_features.items():
                    if col in temp_running_stats and not np.isnan(value):
                        temp_running_stats[col]['sum'] += value
                        temp_running_stats[col]['sum_sq'] += value ** 2
                        temp_running_stats[col]['count'] += 1
            
            if len(X_batch) > 1:
                X_batch = np.array(X_batch)
                
                # Get leaf indices for the batch
                leaf_idx = self.model.apply(X_batch)  # shape: [batch_size, n_trees]
                uniq = len(np.unique(leaf_idx, axis=0))
                total = leaf_idx.shape[0]
                
                print(f"[BATCH_DIVERSITY] Unique leaf-paths in last {total} rows: {uniq}/{total} ({uniq/total*100:.1f}%)")
                
                # Log to file
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_entry = f"[{timestamp}] [BATCH_DIVERSITY] batch_size={total}  leaf_uniq={uniq}/{total}  diversity={uniq/total*100:.1f}%\n"
                
                log_file_path = 'DB/sanity_log.txt'
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
                    
        except Exception as e:
            print(f"Error in log_batch_leaf_diversity: {e}")
    
    def update_memory_with_new_candles(self, df):
        """Update in-memory data with any new candles from the 350-bar fetch"""
        try:
            if self.raw_data.empty:
                print("❌ No existing raw data in memory")
                return False
            
            # Get the latest timestamp we already have
            latest_existing_date = self.raw_data['Date'].max()
            
            # Find any new candles (after our latest timestamp)
            new_candles = df[df['Date'] > latest_existing_date].copy()
            
            if new_candles.empty:
                print(f"No new candles found. Latest in memory: {latest_existing_date}")
                return True
            
            print(f"Found {len(new_candles)} new candles to add to memory")
            
            # Add new candles to raw data
            self.raw_data = pd.concat([self.raw_data, new_candles], ignore_index=True)
            self.raw_data = self.raw_data.sort_values('Date').reset_index(drop=True)
            
            # Recalculate technical indicators for the full updated dataset
            print(f"Recalculating technical indicators with {len(new_candles)} new candles...")
            
            # Prepare data for technical indicators (expects Date as index)
            data_for_indicators = self.raw_data.copy()
            if 'Date' in data_for_indicators.columns:
                data_for_indicators = data_for_indicators.set_index('Date')
            
            self.historical_data = calculate_technical_indicators(data_for_indicators, None)
            
            # Clean up Date ambiguity: ensure Date is ONLY a column, not index
            # calculate_technical_indicators creates both Date index AND Date column, fix this
            if 'Date' in self.historical_data.columns and hasattr(self.historical_data.index, 'name') and self.historical_data.index.name == 'Date':
                # We have both - drop the Date column and reset index to make it a column
                self.historical_data = self.historical_data.drop(columns=['Date']).reset_index()
            elif hasattr(self.historical_data.index, 'name') and self.historical_data.index.name == 'Date':
                # Only index is Date, reset to column
                self.historical_data = self.historical_data.reset_index()
            elif 'Date' not in self.historical_data.columns:
                # No Date column, reset index to create it
                self.historical_data = self.historical_data.reset_index()
                if 'index' in self.historical_data.columns:
                    self.historical_data = self.historical_data.rename(columns={'index': 'Date'})
            
            # Update running statistics with the new candles
            if self.current_features and self.running_stats:
                print(f"Updating running statistics with {len(new_candles)} new candles...")
                
                # Process each new candle to update running stats
                for _, new_candle_row in new_candles.iterrows():
                    # Get the technical indicators for this candle from the updated historical data
                    candle_date = new_candle_row['Date']
                    indicator_row = self.historical_data[self.historical_data['Date'] == candle_date]
                    
                    if not indicator_row.empty:
                        # Preprocess this single row to get features
                        temp_df = indicator_row.copy()
                        try:
                            df_processed, _, _ = preprocess_data(temp_df)
                            if not df_processed.empty:
                                row_features = {col: df_processed.iloc[0][col] for col in self.current_features if col in df_processed.columns}
                                self.update_running_stats(row_features)
                        except Exception as e:
                            print(f"Warning: Could not update running stats for candle {candle_date}: {e}")
            
            # Update last processed timestamp
            self.last_processed_timestamp = self.raw_data['Date'].max()
            
            latest_close = new_candles['Close'].iloc[-1]
            print(f"✅ Added {len(new_candles)} new candles to memory")
            print(f"   Latest: {self.last_processed_timestamp} - Close: ${latest_close:.4f}")
            print(f"   Total bars in memory: {len(self.raw_data):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating memory with new candles: {e}")
            return False
    
    def start_live_trading(self, update_interval=900):  # 15 minutes = 900 seconds
        """Start live trading with periodic updates"""
        if not self.client:
            print("No authenticated client available")
            return
            
        print(f"\nStarting live trading for {self.symbol}...")
        print(f"Update interval: {update_interval} seconds")
        print(f"Trading parameters: {self.trading_params}")
        
        # Sync with account orders first
        print("\nSyncing with account orders...")
        print("DEBUG: About to call sync_with_account_orders()")
        sync_result = self.sync_with_account_orders()
        print(f"DEBUG: sync_with_account_orders() returned: {sync_result}")
        
        # Start the order monitoring thread
        self.start_order_monitor()
        
        # Initialize model (uses full in-memory dataset)
        self.initialize_model()
        
        print("\nLive trading started. Press Ctrl+C to stop.")
        print(f"Data updates: Every 15 minutes")
        print(f"Model retraining: Every {self.retrain_interval * 15} minutes")
        
        try:
            while True:
                # Check if it's time to update data and make predictions
                if self.should_update_data():
                    print(f"\n--- Data Update at {datetime.now().strftime('%H:%M:%S')} ---")
                    
                    # Get latest data
                    candles = self.get_multiple_candles(
                        product_id=self.symbol,
                        target_candles=350,
                        granularity='FIFTEEN_MINUTE'
                    )
                    
                    if candles:
                        df = convert_coinbase_candles_to_dataframe(candles)
                        if not df.empty:
                            # Convert to proper format for memory update
                            df = df.reset_index()
                            df = df.rename(columns={'index': 'Date'})
                            
                            # Update in-memory data with any new candles (could be 1 or more if we missed some)
                            self.update_memory_with_new_candles(df)
                            
                            # Get current price for trading
                            current_price = df['Close'].iloc[-1]
                            print(f"Current price: ${current_price:.4f}")
                            
                            # Update existing trades
                            self.update_trades(current_price)
                            
                            # Update model if needed (trains on last 200 bars from full dataset)
                            self.update_model()
                            
                            # Make prediction (uses full dataset for context)
                            prediction = self.predict_next_bar()
                            if prediction is not None:
                                print(f"Predicted price change: {prediction:.4f}")
                                
                                # Execute new trade if conditions are met
                                self.execute_trade(prediction, current_price)
                                
                                # Print trading summary
                                open_trades_count = self.get_open_trades_count()
                                print(f"Capital: ${self.capital:,.2f}, Open trades: {open_trades_count}")
                                print(f"In-memory data: {len(self.raw_data):,} bars")
                            else:
                                print("No prediction available")
                        else:
                            print("Failed to process data")
                    else:
                        print("Failed to fetch data")
                else:
                    # Not time to update yet, just wait
                    now = datetime.now()
                    current_minute = now.minute
                    next_update_minute = ((current_minute // 15) + 1) * 15
                    if next_update_minute >= 60:
                        next_update_minute = 0
                        next_hour = now.hour + 1
                    else:
                        next_hour = now.hour
                    
                    # Calculate minutes until next update
                    minutes_until_update = (next_update_minute - current_minute) % 15
                    if minutes_until_update == 0:
                        minutes_until_update = 15
                    
                    print(f"Waiting... Next update at {next_hour:02d}:{next_update_minute:02d} (in {minutes_until_update} minutes)")
                
                # Wait 1 minute before checking again
                time.sleep(60)
                    
        except KeyboardInterrupt:
            print("\nStopping live trading...")
            print(f"Final capital: ${self.capital:,.2f}")
            print(f"Total trades: {len(self.trade_history)}")
            if self.trade_history:
                total_profit = sum(trade['profit'] for trade in self.trade_history)
                print(f"Total P&L: ${total_profit:.2f}")
    
    def get_product_candles(self, product_id, start=None, end=None, granularity='FIFTEEN_MINUTE'):
        """
        Fetch historical candle data with retry logic
        
        Args:
            product_id: Trading pair (e.g., 'XRP-USD')
            start: Start time (Unix timestamp)
            end: End time (Unix timestamp)
            granularity: Candle size (FIFTEEN_MINUTE, ONE_HOUR, etc.)
        """
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
                    
                print(f"Requesting candles with params: {params} (attempt {attempt + 1}/{max_retries})")
                candles_response = self.client.get_candles(**params)
     
                # Handle the response object
                if hasattr(candles_response, 'candles'):
                    return candles_response.candles
                elif hasattr(candles_response, '__dict__'):
                    return candles_response.__dict__.get('candles', [])
                else:
                    return candles_response
                    
            except Exception as e:
                print(f"Error fetching candles (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached, giving up")
                    return None
    
    def get_multiple_candles(self, product_id, target_candles=350, granularity='FIFTEEN_MINUTE'):
        """
        Fetch candles (API limit is 350, so we use 350 to be safe)
        
        Args:
            product_id: Trading pair (e.g., 'XRP-USD')
            target_candles: Number of candles to fetch (max 350 due to API limit)
            granularity: Candle size
        """
        all_candles = []
        end_time = datetime.now(timezone.utc)
        
        # Calculate time range for target candles
        # For 15-minute candles, each candle is 15 minutes
        minutes_per_candle = 15 if granularity == 'FIFTEEN_MINUTE' else 60
        total_minutes = target_candles * minutes_per_candle
        
        start_time = end_time - timedelta(minutes=total_minutes)
        
        # Format timestamps as Unix timestamps
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        print(f"Fetching {target_candles} candles from {start_time.strftime('%Y-%m-%dT%H:%M:%SZ')} to {end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        print(f"Unix timestamps: {start_timestamp} to {end_timestamp}")
        
        try:
            candles = self.get_product_candles(
                product_id=product_id,
                start=start_timestamp,
                end=end_timestamp,
                granularity=granularity
            )
            
            if candles:
                print(f"Fetched {len(candles)} candles")
                return candles
            else:
                print("No candles returned")
                return []
                
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return []

def convert_coinbase_candles_to_dataframe(candles_data):
    """
    Convert Coinbase candle data to DataFrame format expected by calculate_technical_indicators
    
    Args:
        candles_data: List of candle objects from Coinbase API
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    if not candles_data:
        return pd.DataFrame()
    
    # Convert the data to a list of dictionaries
    data_list = []
    
    for candle in candles_data:
        # Extract data from candle object
        data_list.append({
            'Open': float(candle.open),
            'High': float(candle.high),
            'Low': float(candle.low),
            'Close': float(candle.close),
            'Volume': float(candle.volume)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Create datetime index from timestamps
    timestamps = [candle.start for candle in candles_data]
    df.index = pd.to_datetime(timestamps, unit='s')  # Convert from Unix timestamp
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_index()
    
    print(f"Converted {len(df)} data points from Coinbase API")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def load_cdp_api_key():
    """Load API credentials from CDP JSON file"""
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'settings', 'cdp_api_key.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract API key name and private key
        api_key = data.get('name')
        api_secret = data.get('privateKey')
        
        if not api_key or not api_secret:
            print("Error: Missing 'name' or 'privateKey' in CDP API key file")
            return None, None
        
        return api_key, api_secret
        
    except FileNotFoundError:
        print(f"Error: CDP API key file not found at {json_path}")
        return None, None
    except json.JSONDecodeError:
        print("Error: Invalid JSON in CDP API key file")
        return None, None
    except Exception as e:
        print(f"Error loading CDP API key: {e}")
        return None, None






def main():
    print("=== Coinbase API Data Fetching ===")
    
    # Load API credentials from CDP JSON file
    print("\nLoading API credentials...")
    api_key, api_secret = load_cdp_api_key()
    
    if not api_key or not api_secret:
        print("Failed to load API credentials. Exiting.")
        return
    
    # Initialize Coinbase API client with 2 years of historical data
    print("Initializing API client with historical data loading...")
    years_data = 6#input("Enter number of years of historical data to load (default 2): ").strip()
    try:
        years_data = float(years_data) if years_data else 2.0
    except ValueError:
        years_data = 2.0
    
    api = CoinbaseAPIClient(api_key=api_key, api_secret=api_secret, years_of_data=years_data)
    print(f"Using authenticated API client with {years_data} years of data")
    
    # Load historical data from API into memory
    print("\n1. Loading historical data from API...")
    if not api.load_historical_data_from_api():
        print("❌ Failed to load historical data. Exiting.")
        return
    
    # Sync with account orders
    print("\n2. Syncing with account orders...")
    api.sync_with_account_orders()
    
    # 3. Fetch account information
    print("\n3. Fetching account information...")
    accounts_response = dict(api.get_accounts())
    if accounts_response:
        accounts = accounts_response['accounts']  # Extract accounts from response
        for account in accounts:
            if account['currency'] == 'USD':
                print(f"USD Balance: {account['available_balance']['value']}")
                api.capital = float(account['available_balance']['value'])
    else:
        print("  Failed to fetch accounts")
    
    # 4. Fetch current orders
    print("\n4. Fetching current orders...")
    orders_response = api.get_orders(status='OPEN')
    if orders_response:
        # Handle both dictionary and list responses
        if isinstance(orders_response, dict):
            orders = orders_response.get('orders', [])
        else:
            orders = list(orders_response)
        
        print(f"Found {len(orders)} open orders")
        for order in orders[:3]:  # Show first 3 orders
            if isinstance(order, dict):
                product_id = order.get('product_id', 'Unknown')
                side = order.get('side', 'Unknown')
                size = order.get('base_size', order.get('total_size', '0'))
                price = order.get('price', '0')
                print(f"  - {product_id}: {side} {size} @ {price}")
            else:
                # Get order details using available attributes
                product_id = getattr(order, 'product_id', 'Unknown')
                side = getattr(order, 'side', 'Unknown')
                size = getattr(order, 'base_size', getattr(order, 'total_size', 'Unknown'))
                price = getattr(order, 'price', 'Unknown')
                print(f"  - {product_id}: {side} {size} @ {price}")
    else:
        print("  No open orders found")
    
    # 5. Use in-memory historical data (already loaded and processed)
    print(f"\n5. Using in-memory historical data...")
    print(f"   Raw data: {len(api.raw_data):,} bars")
    print(f"   With indicators: {len(api.historical_data):,} bars")
    print(f"   Date range: {api.raw_data['Date'].min()} to {api.raw_data['Date'].max()}")
    
    # Get current price from the latest data
    current_price = api.raw_data['Close'].iloc[-1]
    print(f"   Current price: ${current_price:.4f}")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"  - Total bars: {len(api.historical_data):,}")
    print(f"  - Price range: ${api.raw_data['Close'].min():.4f} - ${api.raw_data['Close'].max():.4f}")
    print(f"  - Data span: {(api.raw_data['Date'].max() - api.raw_data['Date'].min()).days} days")
    
    # Model will be initialized when live trading starts
    
    # Start live trading immediately
    print(f"\n{'='*50}")
    print("🚀 STARTING LIVE TRADING...")
    print(f"{'='*50}")
    api.start_live_trading()

if __name__ == "__main__":
    main()