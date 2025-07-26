from ib_insync import *
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import time
import json
from GatherData import calculate_technical_indicators
from utils import preprocess_data, calculate_feature_importance
import xgboost as xgb

class IBKRConnection:
    def __init__(self, host='127.0.0.1', port=7496, client_id=1, symbol='TSLA'):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        self.symbol = symbol
        self.model = None
        self.current_features = None
        self.window_size = None
        self.retrain_interval = None
        self.last_retrain_idx = 0
        self.trading_params = None
        self.open_trades = []
        self.trade_history = []
        self.capital = 1000000  # Initial capital
        self.load_optimized_parameters()
        
    def load_optimized_parameters(self):
        """Load the best parameters from optimization file"""
        try:
            with open(f'Parameters/{self.symbol}_Full_Optimization.json', 'r') as f:
                params = json.load(f)
                best_params = params['top_parameters'][0]['parameters']
                self.trading_params = {
                    'min_risk_percentage': best_params['min_risk_percentage'],
                    'max_risk_percentage': best_params['max_risk_percentage'],
                    'risk_scaling_factor': best_params['risk_scaling_factor'],
                    'risk_reward_ratio': best_params['risk_reward_ratio'],
                    'min_predicted_move': best_params['min_predicted_move'],
                    'partial_take_profit': best_params['partial_take_profit'],
                    'min_holding_period': best_params['min_holding_period'],
                    'max_holding_period': best_params['max_holding_period'],
                    'max_concurrent_trades': best_params['max_concurrent_trades'],
                    'window_size': best_params['calculated_window_size'],
                    'retrain_interval': best_params['calculated_retrain_interval']
                }
                print("Loaded optimized parameters successfully")
        except Exception as e:
            print(f"Error loading parameters: {e}")
            # Use default parameters if loading fails
            self.trading_params = {
                'min_risk_percentage': 0.0057,
                'max_risk_percentage': 0.0058,
                'risk_scaling_factor': 1.93,
                'risk_reward_ratio': 1.88,
                'min_predicted_move': 0.01,
                'partial_take_profit': 0.76,
                'min_holding_period': 14,
                'max_holding_period': 21,
                'max_concurrent_trades': 3,
                'window_size': 20000,
                'retrain_interval': 19276
            }
            print("Using default parameters")

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
            
    def sync_with_account(self):
        """Sync program state with actual IBKR account state"""
        if not self.connected:
            print("Not connected to IBKR")
            return False

        try:
            # Get account summary to update capital
            account = self.ib.managedAccounts()[0]
            self.ib.reqAccountSummary()
            time.sleep(1)  # Wait for data
            account_summary = self.ib.accountSummary(account)
            
            # Update capital from account
            for summary in account_summary:
                if summary.tag == 'NetLiquidation':
                    self.capital = float(summary.value)
                    print(f"Updated capital from account: ${self.capital:,.2f}")
                    break

            # Get current positions
            positions = self.ib.positions()
            self.open_trades = []  # Reset open trades

            # Sync with actual positions
            for position in positions:
                if position.contract.symbol == self.symbol:
                    # Get current market data for the position
                    contract = Stock(self.symbol, 'SMART', 'USD')
                    market_data = self.ib.reqMktData(contract)
                    time.sleep(1)  # Wait for data
                    
                    if market_data.last:
                        current_price = market_data.last
                        # Calculate entry price from position
                        entry_price = position.avgCost
                        size = position.position
                        
                        # Calculate stop loss and take profit based on current price
                        predicted_move = abs(self.trading_params['min_predicted_move'])
                        stop_loss = entry_price * (1 - (predicted_move / self.trading_params['risk_reward_ratio']))
                        take_profit = entry_price * (1 + predicted_move)
                        
                        # Add to open trades
                        self.open_trades.append({
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'size': abs(size),
                            'entry_time': datetime.now(),  # We don't know the actual entry time
                            'trade': None  # No trade object since it's an existing position
                        })
                        print(f"Synced existing position: Entry={entry_price:.2f}, Size={size}")

            # Get recent trades and fill details
            trades = self.ib.trades()
            self.trade_history = []  # Reset trade history
            
            # Get fill details for each trade
            for trade in trades:
                if trade.contract.symbol == self.symbol:
                    # Get all fills for this trade
                    fills = trade.fills
                    if fills:
                        # For each fill, create a trade record
                        for fill in fills:
                            # Determine if this is a buy or sell
                            is_buy = trade.order.action == 'BUY'
                            
                            # Get the execution price and quantity
                            execution_price = fill.execution.price
                            quantity = fill.execution.shares
                            
                            # Calculate realized P&L if this is a sell
                            realized_pnl = 0
                            if not is_buy:
                                # Find the corresponding buy fill(s) for this sell
                                buy_fills = [f for f in fills if f.execution.side == 'BOT']
                                if buy_fills:
                                    # Calculate average entry price from buy fills
                                    avg_entry = sum(f.execution.price * f.execution.shares for f in buy_fills) / sum(f.execution.shares for f in buy_fills)
                                    realized_pnl = (execution_price - avg_entry) * quantity
                            
                            trade_record = {
                                'entry_price': execution_price if is_buy else None,
                                'exit_price': execution_price if not is_buy else None,
                                'size': quantity,
                                'profit': realized_pnl,
                                'reason': 'Existing Position',
                                'entry_time': fill.execution.time,
                                'exit_time': fill.execution.time if not is_buy else None,
                                'side': 'BUY' if is_buy else 'SELL'
                            }
                            self.trade_history.append(trade_record)
                            
                            print(f"Synced trade: {'Buy' if is_buy else 'Sell'} {quantity} shares at {execution_price:.2f}")

            print("Successfully synced with account state")
            return True

        except Exception as e:
            print(f"Error syncing with account: {e}")
            return False

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
            
            # Convert to years if more than 365 days
            if trading_days > 365:
                years = trading_days / 365
                duration = f"{int(years)} Y"
                print(f"Calculated duration: {years:.1f} years")
            else:
                duration = f"{int(trading_days)} D"
                print(f"Calculated duration: {trading_days:.1f} trading days")
            
        try:
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
            return None
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None

    def initialize_model(self, df):
        """Initialize the model with historical data"""
        print("\nInitializing model...")
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
        
        # Initialize model
        model_params = {
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
        
        self.model = xgb.XGBRegressor(**model_params)
        
        # Get preprocessed features and target
        X = df_processed[self.current_features]
        y = df_processed[target_cols].values.ravel()
        
        # Train model
        self.model.fit(X, y)
        print("Model initialized successfully")
        
    def update_model(self, df):
        """Update the model with new data"""
        if len(df) < self.trading_params['window_size']:
            return
            
        # Calculate technical indicators
        df = calculate_technical_indicators(df, None)
        
        # Preprocess data using the utility function
        df_processed, feature_cols, target_cols = preprocess_data(df)
        
        # Recalculate feature importance if needed
        if len(df) % self.trading_params['window_size'] == 0:
            print("\nRecalculating feature importance...")
            self.current_features = calculate_feature_importance(
                df_processed,
                feature_cols,
                target_cols,
                iterations=1,
                save_importance=False,
                visualize_importance=False
            )
        
        # Get preprocessed features and target
        X = df_processed[self.current_features]
        y = df_processed[target_cols].values.ravel()
        
        # Update model
        self.model.fit(X, y)
        print("Model updated successfully")
        
    def predict_next_bar(self, df):
        """Predict the next bar's price change"""
        if self.model is None or len(df) < self.trading_params['window_size']:
            return None
            
        # Calculate technical indicators
        df = calculate_technical_indicators(df, None)
        
        # Preprocess data using the utility function
        df_processed, feature_cols, target_cols = preprocess_data(df)
        
        # Get preprocessed features for prediction
        X = df_processed[self.current_features]
        
        # Make prediction
        prediction = self.model.predict(X.iloc[[-1]])[0]
        return prediction
        
    def execute_trade(self, prediction, current_price):
        """Execute a trade based on prediction"""
        if len(self.open_trades) >= self.trading_params['max_concurrent_trades']:
            return
            
        if prediction < -self.trading_params['min_predicted_move']:
            entry_price = current_price
            
            predicted_move = abs(prediction)
            risk_percentage = min(
                self.trading_params['min_risk_percentage'] * (1 + (predicted_move / self.trading_params['min_predicted_move']) * self.trading_params['risk_scaling_factor']),
                self.trading_params['max_risk_percentage']
            )
            
            risk_amount = self.capital * risk_percentage
            stop_loss = entry_price * (1 - (predicted_move / self.trading_params['risk_reward_ratio']))
            take_profit = entry_price * (1 + predicted_move)
            risk_per_share = entry_price - stop_loss
            
            if risk_per_share <= 0 or np.isnan(risk_per_share):
                return
                
            size = risk_amount / risk_per_share
            size = min(size, self.capital / entry_price)
            size = np.floor(size)
            
            if size <= 0:
                return
                
            # Place the order
            contract = Stock(self.symbol, 'SMART', 'USD')
            order = MarketOrder('BUY', int(size))
            trade = self.ib.placeOrder(contract, order)
            
            self.open_trades.append({
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': size,
                'entry_time': datetime.now(),
                'trade': trade
            })
            
            print(f"Opened new trade: Entry={entry_price:.2f}, Stop={stop_loss:.2f}, Target={take_profit:.2f}, Size={size}")
            
    def update_trades(self, current_price):
        """Update open trades and check for exits"""
        for trade in self.open_trades[:]:
            # Check stop loss
            if current_price <= trade['stop_loss']:
                self.close_trade(trade, current_price, 'Stop Loss')
                continue
                
            # Check take profit
            if current_price >= trade['take_profit']:
                self.close_trade(trade, current_price, 'Take Profit')
                continue
                
            # Check partial take profit
            holding_period = (datetime.now() - trade['entry_time']).total_seconds() / 60
            if holding_period >= self.trading_params['min_holding_period']:
                projected_tp = trade['take_profit']
                projected_entry = trade['entry_price']
                tp_partial = projected_entry + self.trading_params['partial_take_profit'] * (projected_tp - projected_entry)
                
                if current_price >= tp_partial:
                    self.close_trade(trade, current_price, 'Partial Take Profit')
                    continue
                    
                if current_price <= projected_entry:
                    self.close_trade(trade, current_price, 'Break Even')
                    continue
                    
            # Check max holding period
            if holding_period >= self.trading_params['max_holding_period']:
                self.close_trade(trade, current_price, 'Max Holding Period')
                
    def close_trade(self, trade, current_price, reason):
        """Close a trade and update capital"""
        # Place the order
        contract = Stock(self.symbol, 'SMART', 'USD')
        order = MarketOrder('SELL', int(trade['size']))
        self.ib.placeOrder(contract, order)
        
        # Calculate profit
        profit = (current_price - trade['entry_price']) * trade['size']
        self.capital += profit
        
        # Record trade
        trade_record = {
            'entry_price': trade['entry_price'],
            'exit_price': current_price,
            'size': trade['size'],
            'profit': profit,
            'reason': reason,
            'entry_time': trade['entry_time'],
            'exit_time': datetime.now()
        }
        self.trade_history.append(trade_record)
        
        # Remove from open trades
        self.open_trades.remove(trade)
        
        print(f"Closed trade: Entry={trade['entry_price']:.2f}, Exit={current_price:.2f}, Profit={profit:.2f}, Reason={reason}")
        
    def start_live_trading(self):
        """Start live trading with real-time updates"""
        if not self.connected:
            print("Not connected to IBKR")
            return
            
        # Sync with account state first
        if not self.sync_with_account():
            print("Failed to sync with account state")
            return
            
        # Get initial historical data
        print("\nFetching initial historical data...")
        df = self.get_historical_data(self.symbol, num_bars=self.trading_params['window_size'])
        if df is None or df.empty:
            print("Failed to get historical data")
            return
            
        # Initialize model
        self.initialize_model(df)
        
        # Subscribe to real-time bars
        contract = Stock(self.symbol, 'SMART', 'USD')
        self.ib.reqRealTimeBars(contract, barSize=15, whatToShow='TRADES', useRTH=True)  # Changed to 15-minute bars
        
        def on_bar_update(bars):
            # Update historical data
            df = self.get_historical_data(self.symbol, num_bars=self.trading_params['window_size'])
            if df is None or df.empty:
                return
                
            # Update model if needed
            if len(df) - self.last_retrain_idx >= self.trading_params['retrain_interval']:
                self.update_model(df)
                self.last_retrain_idx = len(df)
                
            # Get prediction
            prediction = self.predict_next_bar(df)
            if prediction is not None:
                # Update open trades
                self.update_trades(bars.close)
                
                # Execute new trade if conditions are met
                self.execute_trade(prediction, bars.close)
                
        # Set up the callback
        self.ib.barUpdateEvent += on_bar_update
        
        print("\nLive trading started. Press Ctrl+C to stop.")
        try:
            self.ib.run()
        except KeyboardInterrupt:
            print("\nStopping live trading...")
            self.ib.cancelRealTimeBars(contract)
            self.disconnect()
            
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

# Example usage
if __name__ == "__main__":
    # Connect to TWS and start live trading
    ib = IBKRConnection()
    if ib.connect():
        try:
            ib.start_live_trading()
        finally:
            ib.disconnect()