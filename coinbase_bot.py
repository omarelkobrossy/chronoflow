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
    def __init__(self, api_key=None, api_secret=None, symbol='XRP-USD'):
        """
        Initialize Coinbase API client using official SDK
        
        Args:
            api_key: API key name from CDP JSON file
            api_secret: Private key from CDP JSON file
            symbol: Trading pair (default: 'XRP-USD')
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.client = None
        self.model = None
        self.current_features = None
        self.window_size = 200
        
        # Incremental statistics for expanding standardization
        self.running_stats = {}  # Will store {feature: {'sum': x, 'sum_sq': x, 'count': n}}
        self.last_processed_timestamp = None
        self.retrain_interval = 15  # Number of 15-minute intervals between retrains (15 * 15 = 225 minutes)
        self.last_retrain_idx = 0
        self.last_retrain_time = None
        self.last_update_time = None
        self.trading_params = {
            'min_risk_percentage': 0.008896723760156468,
            'max_risk_percentage': 0.013279057376573214,
            'risk_scaling_factor': 1.7429109176364634,#1.6949727011806939,
            'risk_reward_ratio': 3,#1.6037213842177254,
            'min_predicted_move': 0.003113433706915217,#0.005113433706915217,
            'partial_take_profit': 0.7182278203996404,
            'min_holding_period': 5,
            'max_holding_period': 6,
            'max_concurrent_trades': 8
        }
        self.trade_history = []
        self.capital = 0  # Will be updated from account balance
        self.order_monitor_thread = None
        self.stop_monitoring = False
        self.active_brackets = {}  # Track active bracket orders
        self.model_params = {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1,
            'min_child_weight': 10,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
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
        
        # Load data with historical buffer for proper technical indicator calculation
        if df is None:
            df = self.load_data_with_historical_buffer(buffer_bars=1000, current_bars=350)
        else:
            # If df is provided but might not have enough historical context, reload with buffer
            if len(df) < 500:  # If less than reasonable buffer, reload
                print("Provided DataFrame too small, loading with historical buffer...")
                df = self.load_data_with_historical_buffer(buffer_bars=1000, current_bars=350)
        
        if df.empty:
            print("No data available for model initialization")
            return
            
        print(f"Using {len(df)} bars for model initialization (includes historical buffer)")
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df, None)
        
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
        
        # Initialize running statistics from the full historical dataset
        print("Loading running statistics from full dataset...")
        if not self.initialize_running_stats_from_csv():
            print("Warning: Failed to initialize running stats, using current data only")
            # Fallback to current data if CSV loading fails
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
        
        # Process each row sequentially to build training data with incremental standardization
        for i in range(len(df_processed)):
            row = df_processed.iloc[i]
            row_features = {col: row[col] for col in self.current_features if col in df_processed.columns}
            
            # Get standardized features using current running stats
            if i > 0:  # Skip first row as we need previous data for standardization
                standardized_row = self.get_expanding_standardization_with_stats(row_features, temp_running_stats)
                X_initial.append([standardized_row.get(col, 0.0) for col in self.current_features])
                y_initial.append(row[target_cols[0]])  # Assuming single target
            
            # Update temp running stats with current row
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
        print("Model initialized successfully")
        
    def update_model(self, df=None):
        """Update the model with new data"""
        # Check if it's time to retrain
        if not self.should_retrain_model():
            return
            
        retrain_minutes = self.retrain_interval * 15
        print(f"\nRetraining model at {datetime.now().strftime('%H:%M:%S')} (every {retrain_minutes} minutes)...")
        
        # Load data with historical buffer for proper technical indicator calculation
        if df is None or len(df) < 500:
            print("Loading data with historical buffer for model retraining...")
            df = self.load_data_with_historical_buffer(buffer_bars=1000, current_bars=350)
        
        if df.empty or len(df) < self.window_size:
            print("Insufficient data for model retraining")
            return
            
        print(f"Using {len(df)} bars for model retraining (includes historical buffer)")
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df, None)
        
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
        
        # Process each row sequentially to build training data with incremental standardization
        for i in range(len(df_processed)):
            row = df_processed.iloc[i]
            row_features = {col: row[col] for col in self.current_features if col in df_processed.columns}
            
            # Get standardized features using current running stats
            if i > 0:  # Skip first row as we need previous data for standardization
                standardized_row = self.get_expanding_standardization_with_stats(row_features, temp_running_stats)
                X.append([standardized_row.get(col, 0.0) for col in self.current_features])
                y.append(row[target_cols[0]])  # Assuming single target
            
            # Update temp running stats with current row
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
        print("Model updated successfully")
        
    def predict_next_bar(self, df=None):
        """Predict the next bar's price change using incremental standardization"""
        if self.model is None:
            return None
            
        # Load data with historical buffer for proper technical indicator calculation
        if df is None or len(df) < 500:
            print("Loading data with historical buffer for prediction...")
            df = self.load_data_with_historical_buffer(buffer_bars=1000, current_bars=350)
        
        if df.empty or len(df) < self.window_size:
            print("Insufficient data for prediction")
            return None
            
        # Calculate technical indicators for latest data
        df = calculate_technical_indicators(df, None)
        
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
            # Use a fixed percentage for stop loss, then calculate take profit based on risk_reward_ratio
            stop_loss_percentage = 0.01  # 1% stop loss from entry price
            stop_loss = entry_price * (1 - stop_loss_percentage)
            take_profit = entry_price * (1 + (stop_loss_percentage * self.trading_params['risk_reward_ratio']))
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
    
    def load_data_with_historical_buffer(self, csv_file_path='DB/XRP_USD_fifteenminute_historical.csv', buffer_bars=1000, current_bars=350):
        """
        Load data with historical buffer for proper technical indicator calculation.
        Returns DataFrame with buffer_bars + current_bars, where the last current_bars are the most recent.
        
        Args:
            csv_file_path: Path to historical data CSV
            buffer_bars: Number of historical bars for indicator calculation (default 1000)
            current_bars: Number of current bars for analysis (default 350)
        """
        try:
            print(f"Loading data with {buffer_bars}-bar historical buffer + {current_bars} current bars...")
            
            # Try to load from CSV first
            try:
                historical_df = pd.read_csv(csv_file_path)
                historical_df['Date'] = pd.to_datetime(historical_df['Date'])
                historical_df = historical_df.set_index('Date').sort_index()
                print(f"Loaded {len(historical_df)} historical bars from CSV")
                
                # Get the most recent buffer_bars + current_bars
                total_needed = buffer_bars + current_bars
                if len(historical_df) >= total_needed:
                    buffered_df = historical_df.tail(total_needed).copy()
                    # Reset index to make Date a column instead of index
                    buffered_df = buffered_df.reset_index()
                    print(f"Using {len(buffered_df)} bars from CSV ({buffer_bars} buffer + {current_bars} current)")
                    return buffered_df
                else:
                    print(f"CSV has only {len(historical_df)} bars, need {total_needed}. Using all available from CSV.")
                    buffered_df = historical_df.copy()
                    
                    # Fetch additional bars from API to reach total_needed
                    remaining_needed = total_needed - len(historical_df)
                    print(f"Fetching {remaining_needed} additional bars from API...")
                    
                    # Get the oldest date in CSV to know where to start fetching
                    oldest_csv_date = historical_df.index.min()
                    
                    # Fetch older data
                    end_time = oldest_csv_date
                    start_time = end_time - timedelta(minutes=remaining_needed * 15)
                    
                    older_candles = self.get_product_candles(
                        product_id=self.symbol,
                        start=int(start_time.timestamp()),
                        end=int(end_time.timestamp()),
                        granularity='FIFTEEN_MINUTE'
                    )
                    
                    if older_candles:
                        older_df = convert_coinbase_candles_to_dataframe(older_candles)
                        if not older_df.empty:
                            # Combine older data with CSV data
                            buffered_df = pd.concat([older_df, historical_df], ignore_index=False)
                            buffered_df = buffered_df.sort_index().drop_duplicates()
                            buffered_df = buffered_df.tail(total_needed)
                            print(f"Combined with API data: {len(buffered_df)} total bars")
                    
                    # Reset index to make Date a column instead of index
                    buffered_df = buffered_df.reset_index()
                    return buffered_df
                    
            except FileNotFoundError:
                print("No CSV file found, fetching all data from API...")
                
                # Fetch all needed data from API
                total_needed = buffer_bars + current_bars
                candles = self.get_multiple_candles(
                    product_id=self.symbol,
                    target_candles=min(total_needed, 350),  # API limit
                    granularity='FIFTEEN_MINUTE'
                )
                
                if candles:
                    df = convert_coinbase_candles_to_dataframe(candles)
                    if not df.empty:
                        # Reset index to make Date a column instead of index
                        df = df.reset_index()
                        df = df.rename(columns={'index': 'Date'})
                        print(f"Fetched {len(df)} bars from API")
                        return df
                
                print("Failed to fetch data from API")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error loading data with buffer: {e}")
            return pd.DataFrame()
    
    def initialize_running_stats_from_csv(self, csv_file_path='DB/XRP_USD_fifteenminute_historical.csv'):
        """Initialize running statistics from the full historical dataset"""
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
    
    def update_csv_with_latest_candle(self, df, csv_file_path='DB/XRP_USD_fifteenminute_historical.csv'):
        """Update CSV file with the latest candle from the DataFrame"""
        try:
            # Get the latest candle (last row)
            latest_candle = df.iloc[[-1]].copy()
            
            # Add Date column if it's not already there
            if 'Date' not in latest_candle.columns:
                latest_candle = latest_candle.reset_index()
                latest_candle = latest_candle.rename(columns={'index': 'Date'})
            
            latest_date = latest_candle['Date'].iloc[0] if 'Date' in latest_candle.columns else latest_candle.index[0]
            latest_close = latest_candle['Close'].iloc[0]
            
            # Load existing data
            try:
                existing_df = pd.read_csv(csv_file_path)
                existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                
                # Check if this candle already exists
                if latest_date in existing_df['Date'].values:
                    print(f"Candle for {latest_date} already exists, skipping CSV update...")
                    return True
                
                # Add Date column to latest_candle if it's in the index
                if 'Date' not in latest_candle.columns:
                    latest_candle_for_csv = latest_candle.reset_index()
                    latest_candle_for_csv = latest_candle_for_csv.rename(columns={'index': 'Date'})
                else:
                    latest_candle_for_csv = latest_candle.copy()
                
                # Append the new candle
                updated_df = pd.concat([existing_df, latest_candle_for_csv], ignore_index=True)
                updated_df = updated_df.sort_values('Date').reset_index(drop=True)
                
            except FileNotFoundError:
                # Create new file with this candle
                if 'Date' not in latest_candle.columns:
                    updated_df = latest_candle.reset_index()
                    updated_df = updated_df.rename(columns={'index': 'Date'})
                else:
                    updated_df = latest_candle.copy()
            
            # Save updated data
            import os
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
            updated_df.to_csv(csv_file_path, index=False)
            
            # Update last processed timestamp for running stats tracking
            self.last_processed_timestamp = latest_date
            
            print(f"✅ Added new candle to CSV: {latest_date} - Close: ${latest_close:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating CSV with latest candle: {e}")
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
        
        # Initialize model (it will load data with historical buffer automatically)
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
                            # Update CSV file with the latest candle
                            self.update_csv_with_latest_candle(df)
                            
                            # Get current price for trading
                            current_price = df['Close'].iloc[-1]
                            print(f"Current price: ${current_price:.4f}")
                            
                            # Update existing trades
                            self.update_trades(current_price)
                            
                            # Update model if needed (will load data with buffer internally)
                            self.update_model()
                            
                            # Make prediction (will load data with buffer internally)
                            prediction = self.predict_next_bar()
                            if prediction is not None:
                                print(f"Predicted price change: {prediction:.4f}")
                                
                                # Execute new trade if conditions are met
                                self.execute_trade(prediction, current_price)
                                
                                # Print trading summary
                                open_trades_count = self.get_open_trades_count()
                                print(f"Capital: ${self.capital:,.2f}, Open trades: {open_trades_count}")
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

def analyze_csv_gaps(csv_file_path='DB/XRP_USD_fifteenminute_historical.csv'):
    """
    Analyze CSV file to detect all time gaps and return gap information
    """
    try:
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        print(f"=== Gap Analysis for {csv_file_path} ===")
        print(f"Total rows: {len(df)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Calculate time differences between consecutive rows
        df['time_diff'] = df['Date'].diff()
        
        # Find gaps larger than 15 minutes (missing candles)
        gaps = df[df['time_diff'] > pd.Timedelta(minutes=16)].copy()
        
        if len(gaps) == 0:
            print("✅ No gaps found - data is continuous!")
            return []
        
        print(f"\n❌ Found {len(gaps)} gaps larger than 15 minutes:")
        
        gap_list = []
        for i, gap_row in gaps.iterrows():
            if i > 0:
                prev_date = df.iloc[i-1]['Date']
                curr_date = gap_row['Date']
                gap_hours = gap_row['time_diff'].total_seconds() / 3600
                missing_candles = int(gap_hours * 4)  # 4 candles per hour (15-min intervals)
                
                gap_info = {
                    'start': prev_date,
                    'end': curr_date,
                    'duration_hours': gap_hours,
                    'missing_candles': missing_candles
                }
                gap_list.append(gap_info)
                
                print(f"  Gap #{len(gap_list)}: {prev_date} -> {curr_date}")
                print(f"    Duration: {gap_hours:.1f} hours ({missing_candles} missing candles)")
        
        return gap_list
        
    except Exception as e:
        print(f"Error analyzing gaps: {e}")
        return []

def fill_all_gaps(csv_file_path='DB/XRP_USD_fifteenminute_historical.csv', max_gaps_to_fill=50):
    """
    Comprehensive gap filling function that detects and fills all gaps in historical data
    """
    print("=== Comprehensive Gap Filling ===")
    
    # Load API credentials
    api_key, api_secret = load_cdp_api_key()
    if not api_key or not api_secret:
        print("Failed to load API credentials. Exiting.")
        return False
    
    # Initialize API client
    api = CoinbaseAPIClient(api_key=api_key, api_secret=api_secret)
    
    # Analyze existing gaps
    gaps = analyze_csv_gaps(csv_file_path)
    if not gaps:
        print("No gaps to fill!")
        return True
    
    print(f"\nFound {len(gaps)} gaps to fill (processing up to {max_gaps_to_fill})")
    
    # Load existing data
    df = pd.read_csv(csv_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    filled_gaps = 0
    total_bars_added = 0
    
    # Process gaps from newest to oldest to maintain data integrity
    for i, gap in enumerate(sorted(gaps, key=lambda x: x['start'], reverse=True)[:max_gaps_to_fill]):
        print(f"\n🔧 Filling gap #{i+1}/{min(len(gaps), max_gaps_to_fill)}")
        print(f"   From: {gap['start']}")
        print(f"   To: {gap['end']}")
        print(f"   Duration: {gap['duration_hours']:.1f} hours")
        
        # Calculate how many API requests needed
        max_bars_per_request = 300  # Safe limit
        minutes_per_bar = 15
        total_minutes_needed = gap['duration_hours'] * 60
        bars_needed = int(total_minutes_needed / minutes_per_bar)
        requests_needed = max(1, (bars_needed // max_bars_per_request) + 1)
        
        print(f"   Need ~{bars_needed} bars ({requests_needed} API requests)")
        
        if requests_needed > 20:  # Safety limit
            print(f"   ⚠️  Gap too large ({requests_needed} requests), skipping...")
            continue
        
        # Fetch data to fill this gap
        gap_data = []
        current_end = gap['end']
        
        for req_num in range(min(requests_needed, 20)):
            # Calculate start time for this request
            minutes_this_request = min(max_bars_per_request * minutes_per_bar, 
                                     total_minutes_needed - req_num * max_bars_per_request * minutes_per_bar)
            current_start = current_end - timedelta(minutes=minutes_this_request)
            
            # Don't go before the gap start
            if current_start < gap['start']:
                current_start = gap['start']
            
            print(f"     Request {req_num+1}: {current_start} to {current_end}")
            
            try:
                gap_candles = api.get_product_candles(
                    product_id='XRP-USD',
                    start=int(current_start.timestamp()),
                    end=int(current_end.timestamp()),
                    granularity='FIFTEEN_MINUTE'
                )
                
                if gap_candles:
                    gap_df = convert_coinbase_candles_to_dataframe(gap_candles)
                    if not gap_df.empty:
                        gap_df = gap_df.reset_index()
                        gap_df = gap_df.rename(columns={'index': 'Date'})
                        # Filter to only data within the gap period
                        gap_df = gap_df[(gap_df['Date'] > gap['start']) & (gap_df['Date'] < gap['end'])]
                        if not gap_df.empty:
                            gap_data.append(gap_df)
                            print(f"       Got {len(gap_df)} bars")
                        else:
                            print(f"       No data in gap period")
                    else:
                        print(f"       Failed to convert candle data")
                else:
                    print(f"       No candles returned")
                
                # Update for next iteration
                current_end = current_start
                if current_start <= gap['start']:
                    break
                    
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"       Error fetching data: {e}")
                continue
        
        # Combine gap data and add to main dataframe
        if gap_data:
            combined_gap_df = pd.concat(gap_data, ignore_index=True)
            combined_gap_df = combined_gap_df.sort_values('Date').drop_duplicates(subset=['Date'], keep='first')
            
            # Add to main dataframe
            df = pd.concat([df, combined_gap_df], ignore_index=True)
            df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)
            
            bars_added = len(combined_gap_df)
            total_bars_added += bars_added
            filled_gaps += 1
            
            print(f"   ✅ Added {bars_added} bars to fill gap")
        else:
            print(f"   ❌ Failed to get data for this gap")
    
    # Save updated data
    if filled_gaps > 0:
        df.to_csv(csv_file_path, index=False)
        print(f"\n🎉 Gap filling complete!")
        print(f"   Filled {filled_gaps} gaps")
        print(f"   Added {total_bars_added} total bars")
        print(f"   Updated file: {csv_file_path}")
        
        # Re-analyze to show improvement
        print("\n📊 Post-fill analysis:")
        remaining_gaps = analyze_csv_gaps(csv_file_path)
        if len(remaining_gaps) < len(gaps):
            print(f"✅ Reduced gaps from {len(gaps)} to {len(remaining_gaps)}")
        else:
            print(f"⚠️  Still have {len(remaining_gaps)} gaps remaining")
    else:
        print("\n❌ No gaps were successfully filled")
    
    return filled_gaps > 0

def update_historical_data(csv_file_path='DB/XRP_USD_fifteenminute_historical.csv'):
    """
    Enhanced function to update historical data with comprehensive gap detection and filling
    """
    print("=== Enhanced Historical Data Update ===")
    
    # First, fill any existing gaps in the historical data
    print("Step 1: Analyzing and filling existing gaps...")
    fill_all_gaps(csv_file_path)
    
    # Then proceed with normal update (add latest data)
    print("\nStep 2: Adding latest data...")
    
    # Load API credentials
    api_key, api_secret = load_cdp_api_key()
    if not api_key or not api_secret:
        print("Failed to load API credentials. Exiting.")
        return
    
    # Initialize API client
    api = CoinbaseAPIClient(api_key=api_key, api_secret=api_secret)
    
    # Load existing data
    try:
        existing_df = pd.read_csv(csv_file_path)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        existing_df = existing_df.sort_values('Date').reset_index(drop=True)
        last_date = existing_df['Date'].max()
        print(f"Loaded existing data: {len(existing_df)} rows, last date: {last_date}")
    except FileNotFoundError:
        print("No existing data file found. Will create new one.")
        existing_df = pd.DataFrame()
        last_date = None
    
    # Get latest 350 bars
    print("Fetching latest 350 bars...")
    candles = api.get_multiple_candles(
        product_id='XRP-USD',
        target_candles=350,
        granularity='FIFTEEN_MINUTE'
    )
    
    if not candles:
        print("Failed to fetch candle data")
        return
    
    # Convert to DataFrame
    new_df = convert_coinbase_candles_to_dataframe(candles)
    if new_df.empty:
        print("Failed to convert candle data")
        return
    
    # Add Date column (convert index to column)
    new_df = new_df.reset_index()
    new_df = new_df.rename(columns={'index': 'Date'})
    
    print(f"Fetched new data: {len(new_df)} rows, range: {new_df['Date'].min()} to {new_df['Date'].max()}")
    
    # Check for gaps and fetch more data if needed
    if last_date is not None:
        gap_start = last_date
        gap_end = new_df['Date'].min()
        gap_hours = (gap_end - gap_start).total_seconds() / 3600
        
        if gap_hours > 24:  # If gap is more than 24 hours, try to fill it
            print(f"\nDetected {gap_hours:.1f} hour gap. Attempting to fill...")
            
            # Calculate how many 350-bar requests we need
            bars_per_request = 300  # Safe limit
            minutes_per_bar = 15
            minutes_per_request = bars_per_request * minutes_per_bar
            gap_minutes = gap_hours * 60
            requests_needed = int(gap_minutes / minutes_per_request) + 1
            
            print(f"Need approximately {requests_needed} requests to fill gap")
            
            # Fetch data to fill the gap
            fill_data = []
            current_end = gap_end
            
            for i in range(min(requests_needed, 10)):  # Limit to 10 requests max
                current_start = current_end - timedelta(minutes=minutes_per_request)
                if current_start <= gap_start:
                    current_start = gap_start + timedelta(minutes=15)  # Avoid overlap
                
                print(f"  Fetching gap data {i+1}/{min(requests_needed, 10)}: {current_start} to {current_end}")
                
                gap_candles = api.get_product_candles(
                    product_id='XRP-USD',
                    start=int(current_start.timestamp()),
                    end=int(current_end.timestamp()),
                    granularity='FIFTEEN_MINUTE'
                )
                
                if gap_candles:
                    gap_df = convert_coinbase_candles_to_dataframe(gap_candles)
                    if not gap_df.empty:
                        gap_df = gap_df.reset_index()
                        gap_df = gap_df.rename(columns={'index': 'Date'})
                        fill_data.append(gap_df)
                        print(f"    Got {len(gap_df)} bars")
                
                current_end = current_start
                if current_start <= gap_start:
                    break
            
            # Combine fill data
            if fill_data:
                fill_df = pd.concat(fill_data, ignore_index=True)
                fill_df = fill_df.sort_values('Date').drop_duplicates(subset=['Date'], keep='first')
                new_df = pd.concat([fill_df, new_df], ignore_index=True)
                print(f"Added {len(fill_df)} bars to fill gap")
    
    # Combine with existing data
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Remove duplicates and sort
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    print(f"\nCombined data: {initial_count} -> {len(combined_df)} rows (removed {initial_count - len(combined_df)} duplicates)")
    print(f"Final date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    
    # Save updated data
    import os
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    combined_df.to_csv(csv_file_path, index=False)
    print(f"Saved updated data to {csv_file_path}")
    
    # Final gap analysis
    print("\nStep 3: Final gap analysis...")
    analyze_csv_gaps(csv_file_path)
    
    return combined_df

def main():
    print("=== Coinbase API Data Fetching ===")
    
    # Load API credentials from CDP JSON file
    print("\nLoading API credentials...")
    api_key, api_secret = load_cdp_api_key()
    
    if not api_key or not api_secret:
        print("Failed to load API credentials. Exiting.")
        return
    
    # Initialize Coinbase API client
    api = CoinbaseAPIClient(api_key=api_key, api_secret=api_secret)
    print("Using authenticated API client")
    
    # Sync with account orders
    print("\nSyncing with account orders...")
    api.sync_with_account_orders()
    
    # 1. Fetch account information
    print("\n1. Fetching account information...")
    accounts_response = dict(api.get_accounts())
    #print(accounts_response)
    if accounts_response:
        accounts = accounts_response['accounts']  # Extract accounts from response
        #print(accounts)
        for account in accounts:
            if account['currency'] == 'USD':
                print(f"USD Balance: {account['available_balance']['value']}")
                api.capital = float(account['available_balance']['value'])
    else:
        print("  Failed to fetch accounts")
    
    # 2. Fetch current orders
    print("\n2. Fetching current orders...")
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
    
    # 3. Fetch last 400 bars for XRP/USD (15-minute candles)
    print("\n3. Fetching XRP/USD historical data...")
    
    candles = api.get_multiple_candles(
        product_id='XRP-USD',
        target_candles=350,  # Reduced to stay within API limit
        granularity='FIFTEEN_MINUTE'
    )
    
    if candles:
        # Handle the response object properly
        if hasattr(candles, 'candles'):
            candles_list = candles.candles
        elif hasattr(candles, '__dict__'):
            candles_list = candles.__dict__.get('candles', [])
        else:
            candles_list = list(candles)
        
        print(f"Fetched {len(candles_list)} candles for XRP/USD")
        
        # Convert to DataFrame
        df = convert_coinbase_candles_to_dataframe(candles_list)
        
        if not df.empty:
            # Calculate technical indicators
            print("\n4. Calculating technical indicators...")
            df_with_indicators = calculate_technical_indicators(df)
            
            # Save the results
            output_file = "XRP_USD_15min_coinbase_indicators.csv"
            df_with_indicators.to_csv(output_file)
            print(f"Saved {len(df_with_indicators)} data points to {output_file}")
            
            # Print sample of the data
            print("\nFirst few rows of the data:")
            print(df_with_indicators.head())
            
            # Print date range
            print(f"\nDate range: {df_with_indicators.index.min()} to {df_with_indicators.index.max()}")
            
            # Print summary statistics
            print(f"\nSummary statistics:")
            print(f"  - Total bars: {len(df_with_indicators)}")
            print(f"  - Price range: ${df_with_indicators['Close'].min():.4f} - ${df_with_indicators['Close'].max():.4f}")
            print(f"  - Current price: ${df_with_indicators['Close'].iloc[-1]:.4f}")
            
            # Initialize model for trading
            print("\n5. Initializing trading model...")
            api.initialize_model(df_with_indicators)
            
            # Get current price for trading
            current_price = df_with_indicators['Close'].iloc[-1]
            print(f"Current price: ${current_price:.4f}")
            
            # Make prediction (for demonstration only)
            print("\n6. Making trading prediction...")
            prediction = api.predict_next_bar(df_with_indicators)
            if prediction is not None:
                print(f"Predicted price change: {prediction:.4f}")
                
                # Print current status (no trading yet)
                print(f"\nCurrent Status:")
                print(f"  - Capital: ${api.capital:,.2f}")
                open_trades_count = api.get_open_trades_count()
                print(f"  - Open trades: {open_trades_count}")
                print(f"  - Trade history: {len(api.trade_history)} trades")
                
                if open_trades_count > 0:
                    print(f"  - Active orders on account: {open_trades_count}")
                
                if api.trade_history:
                    total_profit = sum(trade['profit'] for trade in api.trade_history)
                    print(f"  - Total realized P&L: ${total_profit:.2f}")
            else:
                print("No prediction available")
            
            # Ask if user wants to start live trading
            print("\n" + "="*50)
            print("TRADING OPTIONS:")
            print("1. Start live trading (continuous updates)")
            print("2. Update historical data only") 
            print("3. Analyze CSV gaps only")
            print("4. Fill all CSV gaps")
            print("5. Exit")
            choice = input("\nEnter your choice (1, 2, 3, 4, or 5): ").strip()
            
            if choice == "1":
                print("\nStarting live trading...")
                api.start_live_trading()
            elif choice == "2":
                print("\nUpdating historical data...")
                update_historical_data()
            elif choice == "3":
                print("\nAnalyzing CSV gaps...")
                analyze_csv_gaps()
            elif choice == "4":
                print("\nFilling all CSV gaps...")
                fill_all_gaps()
            else:
                print("Exiting...")
            
        else:
            print("Error: No data to process")
    else:
        print("Error: Failed to fetch candle data")

if __name__ == "__main__":
    import sys
    
    # Check if user wants to just update data
    if len(sys.argv) > 1 and sys.argv[1] == "--update-data":
        update_historical_data()
    else:
        main()