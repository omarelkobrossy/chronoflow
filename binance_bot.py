# Binance API integration using WebSocket kline streams
# Uses Binance API key from settings/binance_api_key.json

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
import websocket
import ssl
from binance.spot import Spot
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from utils import send_telegram_message, get_filters, fmt_qty, fmt_price, floor_to_step

# Add the Quant directory to the path to import from GatherData.py
sys.path.append(os.path.join(os.path.dirname(__file__), 'Quant'))
from GatherData import calculate_technical_indicators
from utils import preprocess_data, calculate_feature_importance, make_ref_bins, psi_from_bins

class BinanceAPIClient:
    def __init__(self, api_key=None, api_secret=None, symbol='XRP/USDC', years_of_data=2):
        """
        Initialize Binance API client using WebSocket streams
        
        Args:
            api_key: API key from Binance
            api_secret: API secret from Binance
            symbol: Trading pair (default: 'XRPUSDC')
            years_of_data: Number of years of historical data to load (default: 2)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol.replace('/', '')
        self.symbol_base, self.symbol_quote = symbol.split('/')
        self.years_of_data = years_of_data
        self.client = None
        self.ws_stream_client = None  # SpotWebsocketStreamClient
        self.ws_api_client = None  # SpotWebsocketAPIClient
        self.model = None
        self.current_features = None
        self.maker_fee = 0.001  # Will be updated from API
        self.taker_fee = 0.001  # Will be updated from API

        # In-memory data storage
        self.historical_data = pd.DataFrame()  # Full historical dataset with indicators
        self.raw_data = pd.DataFrame()  # Raw OHLCV data without indicators
        
        # Incremental statistics for expanding standardization
        self.running_stats = {}  # Will store {feature: {'sum': x, 'sum_sq': x, 'count': n}}
        self.last_processed_timestamp = None
        self.trading_params = {
            "min_risk_percentage": 0.166116101686702,
            "max_risk_percentage": 0.72910312515355,
            "risk_scaling_factor": 2.810894735034883,
            "risk_reward_ratio": 1.9475937758684003,
            "min_predicted_move": 0.005249181572122793,
            "partial_take_profit": 0.8158426307375336,
            "min_holding_period": 20,
            "max_holding_period": 63,
            "max_concurrent_trades": 7,
            "stop_loss_atr_multiplier": 1.1392661059812488,
            "atr_predicted_weight": 0.27595429613948874,
            "aggressiveness": 3.478895668999351,
            "feature_count_k": 26,
            "window_size": 33880
        }
        self.window_size = self.trading_params['window_size']
        
        # PSI-based drift detection parameters
        self.psi_params = {
            "psi_bins": 20,
            "psi_feat_count": self.trading_params['feature_count_k'],
            "psi_check_step": 1,  # Check every 1 bars
            "psi_window": self.window_size,  # Same as window_size
            "psi_threshold_hi": 0.25,  # Strong drift threshold
            "psi_threshold_lo": 0.12,  # Mild drift threshold
            "psi_cooldown": 50,  # Min bars between retrains
            "min_segment": 300,  # Minimum segment size
        }
        
        # PSI reference data
        self.psi_ref_data = None
        self.psi_ref_bins = {}
        self.last_retrain_bar = None
        self.segment_start = None
        self.trade_history = []
        self.capital = 0  # Will be updated from account balance
        self.model_params = {
            "learning_rate": 0.08756625379937401,
            "n_estimators": 1366,
            "max_depth": 5,
            "max_leaves": 36,
            "min_child_weight": 0.6894120769740582,
            "gamma": 0.1925456272650492,
            "subsample": 0.9334546855864236,
            "colsample_bytree": 0.4448986278464316,
            "colsample_bylevel": 0.9536809078007669,
            "reg_lambda": 1.844693074360457,
            "reg_alpha": 0.14621304476183838,
            "max_bin": 256,
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'cuda',
        }
        
        # WebSocket stream management
        self.stream_active = False
        self.last_kline_data = None
        self.kline_lock = threading.Lock()
        
        # Symbol info cache for lot size validation
        self.symbol_info = None
        
        # Account info cache
        self.account_info = None
        
        if api_key and api_secret:
            self.client = Spot(api_key=api_key, api_secret=api_secret)
            # Initialize WebSocket clients
            self.ws_stream_client = None  # Will be created when starting stream
            self.ws_api_client = SpotWebsocketAPIClient(api_key=api_key, api_secret=api_secret)
            # Fetch actual trading fees from API
            self.fetch_trading_fees()
            
            # Load symbol info for lot size validation
            self.load_symbol_info()
            
            # Load account info
            self.account_info = self.get_account_info()
    
    def fetch_trading_fees(self):
        """Fetch actual trading fees from Binance API"""
        try:
            if not self.client:
                print("No authenticated client available for fetching fees")
                return
            
            print("Fetching trading fees from Binance API...")
            
            # Get trading fees for the specific symbol
            fee_info = self.client.trade_fee(symbol=self.symbol)
            
            if fee_info:
                # Extract maker and taker fees from the response
                print(fee_info)
                self.maker_fee = float(fee_info['makerCommission'])
                self.taker_fee = float(fee_info['takerCommission'])
                
                print(f"[OK] Trading fees fetched:")
                print(f"   Maker fee: {self.maker_fee:.4f} ({self.maker_fee*100:.2f}%)")
                print(f"   Taker fee: {self.taker_fee:.4f} ({self.taker_fee*100:.2f}%)")
            else:
                print("[WARN] Could not fetch trading fees, using default values")
                print(f"   Maker fee: {self.maker_fee:.4f} ({self.maker_fee*100:.2f}%)")
                print(f"   Taker fee: {self.taker_fee:.4f} ({self.taker_fee*100:.2f}%)")
                
        except Exception as e:
            print(f"[ERROR] Error fetching trading fees: {e}")
            print("Using default fee values")
            print(f"   Maker fee: {self.maker_fee:.4f} ({self.maker_fee*100:.2f}%)")
            print(f"   Taker fee: {self.taker_fee:.4f} ({self.taker_fee*100:.2f}%)")
            print("Note: Fee fetching requires valid API credentials")
    
    def load_symbol_info(self):
        """Load and cache symbol info for lot size validation"""
        try:
            if not self.client:
                print("No authenticated client available for fetching symbol info")
                return
            
            print(f"Loading symbol info for {self.symbol}...")
            
            # Get exchange info
            exchange_info = self.client.exchange_info()
            
            # Find our symbol
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == self.symbol:
                    self.symbol_info = symbol_info
                    print(f"✅ Symbol info loaded for {self.symbol}")
                    
                    # Print lot size requirements
                    lot_size_filter = None
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'LOT_SIZE':
                            lot_size_filter = filter_info
                            break
                    
                    if lot_size_filter:
                        min_qty = float(lot_size_filter['minQty'])
                        max_qty = float(lot_size_filter['maxQty'])
                        step_size = float(lot_size_filter['stepSize'])
                        print(f"   Min quantity: {min_qty}")
                        print(f"   Max quantity: {max_qty}")
                        print(f"   Step size: {step_size}")
                    else:
                        print("   No LOT_SIZE filter found")
                    
                    return
            
            print(f"❌ Symbol {self.symbol} not found in exchange info")
            
        except Exception as e:
            print(f"❌ Error loading symbol info: {e}")
    
    def start_kline_stream(self):
        """Start WebSocket kline stream for real-time data"""
        try:
            print(f"Starting kline stream for {self.symbol}...")
            
            # Start kline stream (15-minute intervals)
            def message_handler(ws, message):
                self.handle_kline_message(message)
            
            # Use the correct method signature for binance-connector
            self.ws_stream_client = SpotWebsocketStreamClient(on_message=message_handler)
            self.ws_stream_client.kline(
                symbol=self.symbol.lower(),
                interval='15m'
            )
            
            self.stream_active = True
            print(f"✅ Kline stream started for {self.symbol}")
            print(f"[WEBSOCKET] Stream active status: {self.stream_active}")
            
        except Exception as e:
            print(f"❌ Error starting kline stream: {e}")
            self.stream_active = False
    
    def handle_kline_message(self, msg):
        """Handle incoming kline data from WebSocket"""
        try:
            # Print heartbeat for WebSocket messages
            # print(f"[WEBSOCKET] Received message at {datetime.now().strftime('%H:%M:%S')}")
            
            with self.kline_lock:
                # Parse JSON message if it's a string
                if isinstance(msg, str):
                    import json
                    msg = json.loads(msg)
                
                # Skip non-kline messages (like ping/pong)
                if 'k' not in msg:
                    return
                
                # Extract kline data
                kline = msg['k']
                
                # Add debugging for WebSocket messages
                # print(f"DEBUG: Received WebSocket message - Symbol: {kline['s']}, Interval: {kline['i']}, Closed: {kline['x']}")
                # print(f"DEBUG: Timestamp: {kline['t']}, Close Time: {kline['T']}")
                # print(f"DEBUG: OHLCV: O={kline['o']}, H={kline['h']}, L={kline['l']}, C={kline['c']}, V={kline['v']}")
                
                # Only process closed candles (is_closed = True)
                if kline['x']:  # x indicates if kline is closed
                    kline_data = {
                        'timestamp': int(kline['t']),
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'close_time': int(kline['T'])
                    }
                    
                    self.last_kline_data = kline_data
                    
                    # Process the new candle
                    self.process_new_candle(kline_data)
                #else:
                    #print(f"DEBUG: Candle not closed yet, ignoring")     
                    
        except Exception as e:
            print(f"❌ Error handling kline message: {e}")
            print(msg)
    
    def process_new_candle(self, kline_data):
        """Process a new closed candle and update model/predictions"""
        try:
            print(f"\n--- New Candle Closed at {datetime.now().strftime('%H:%M:%S')} ---")
            print(f"OHLCV: O={kline_data['open']:.4f}, H={kline_data['high']:.4f}, "
                  f"L={kline_data['low']:.4f}, C={kline_data['close']:.4f}, V={kline_data['volume']:.2f}")
            
            # Convert to DataFrame format with timezone-aware datetime
            new_candle_df = pd.DataFrame([{
                'Date': pd.to_datetime(kline_data['timestamp'], unit='ms', utc=True),
                'Open': kline_data['open'],
                'High': kline_data['high'],
                'Low': kline_data['low'],
                'Close': kline_data['close'],
                'Volume': kline_data['volume']
            }])
            
            # Update in-memory data with new candle
            if self.update_memory_with_new_candle(new_candle_df):
                current_price = kline_data['close']
                
                # Update existing trades
                self.update_trades(current_price)
                
                # Update model if needed (PSI drift detection)
                self.update_model()
                
                # Make prediction
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
            
        except Exception as e:
            print(f"❌ Error processing new candle: {e}")
    
    def stop_kline_stream(self):
        """Stop WebSocket kline stream"""
        try:
            if self.ws_stream_client:
                self.ws_stream_client.stop()
            
            self.stream_active = False
            print("✅ Kline stream stopped")
            
        except Exception as e:
            print(f"❌ Error stopping kline stream: {e}")
    
    def get_account_info(self):
        """Get account information from Binance"""
        try:
            if not self.client:
                print("No authenticated client available")
                return None
            
            account_info = self.client.account()
            return account_info
            
        except Exception as e:
            print(f"Error fetching account info: {e}")
            return None
    
    def refresh_account_info(self):
        """Refresh account info cache"""
        self.account_info = self.get_account_info()
        return self.account_info
    
    def get_usd_balance(self):
        """Get USDC balance from account"""
        try:
            if not self.client:
                print("No authenticated client available")
                return None
            
            account_info = self.get_account_info()
            if not account_info:
                print("Failed to fetch account info")
                return None
            
            # Look for USDC balance
            for balance in account_info['balances']:
                if balance['asset'] == 'USDC':
                    free_balance = float(balance['free'])
                    locked_balance = float(balance['locked'])
                    total_balance = free_balance + locked_balance
                    print(f"USDC Balance: Free={free_balance:.2f}, Locked={locked_balance:.2f}, Total={total_balance:.2f}")
                    return total_balance
            
            print("No USDC balance found")
            return None
            
        except Exception as e:
            print(f"Error fetching USDC balance: {e}")
            return None
    
    def update_capital_from_account(self):
        """Update capital from actual USDC balance"""
        try:
            print("DEBUG: update_capital_from_account() called")
            usdc_balance = self.get_usd_balance()
            print(f"DEBUG: get_usd_balance() returned: {usdc_balance}")
            if usdc_balance is not None:
                self.capital = usdc_balance
                print(f"Updated capital from account: ${self.capital:,.2f}")
                return True
            else:
                print("Failed to update capital from account")
                return False
        except Exception as e:
            print(f"Error updating capital: {e}")
            return False
    
    def get_open_orders(self):
        """Get all open orders from Binance"""
        try:
            if not self.client:
                print("No authenticated client available")
                return []
            
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            return open_orders
            
        except Exception as e:
            print(f"Error fetching open orders: {e}")
            return []
    
    def get_open_trades_count(self):
        """Get count of open trades from account"""
        try:
            if not self.client:
                return 0
            
            print("DEBUG: Getting open orders...")
            open_orders = self.get_open_orders()
            
            count = len(open_orders)
            print(f"DEBUG: Open orders count: {count}")
            return count
            
        except Exception as e:
            print(f"Error getting open trades count: {e}")
            return 0
    
    def sync_with_account_orders(self):
        """Sync with actual open orders from Binance account and update capital"""
        try:
            if not self.client:
                print("No authenticated client available")
                return False
            
            # Update capital from account first
            print("Syncing with account...")
            if not self.update_capital_from_account():
                print("Warning: Failed to update capital from account")
            
            # Get all open orders
            open_orders = self.get_open_orders()
            if not open_orders:
                print("No open orders found")
                return True
            
            print(f"Found {len(open_orders)} open orders on account")
            
            # Filter orders for our symbol
            symbol_orders = [order for order in open_orders if order['symbol'] == self.symbol]
            
            if symbol_orders:
                print(f"Found {len(symbol_orders)} open orders for {self.symbol}")
                for order in symbol_orders:
                    print(f"  - {order['side']} {order['origQty']} {self.symbol} @ {order['price']} ({order['type']}) - ID: {order['orderId']}")
            else:
                print(f"No open orders for {self.symbol}")
            
            return True
            
        except Exception as e:
            print(f"Error syncing with account orders: {e}")
            return False
    
    def execute_trade(self, prediction, current_price):
        """Execute a trade based on prediction"""
        # Check actual open trades from account
        open_trades_count = self.get_open_trades_count()
        if open_trades_count >= self.trading_params['max_concurrent_trades']:
            print(f"Maximum concurrent trades reached ({open_trades_count}/{self.trading_params['max_concurrent_trades']})")
            return
        
        print(f"Prediction: {prediction}, Min Predicted Move: {-self.trading_params['min_predicted_move']}")
        if prediction < -self.trading_params['min_predicted_move'] or True:
            entry_price = current_price
            
            predicted_move = abs(prediction)
            
            # Calculate how much the predicted move exceeds the minimum threshold
            move_excess_ratio = (predicted_move - self.trading_params['min_predicted_move']) / self.trading_params['min_predicted_move']
            
            # Apply aggressiveness scaling: higher aggressiveness = faster scaling to max risk
            scaled_excess = move_excess_ratio ** (1 / self.trading_params['aggressiveness'])
            
            # Calculate risk percentage with aggressiveness-controlled scaling
            risk_percentage = min(
                self.trading_params['min_risk_percentage'] + (self.trading_params['max_risk_percentage'] - self.trading_params['min_risk_percentage']) * min(scaled_excess * self.trading_params['risk_scaling_factor'], 1.0),
                self.trading_params['max_risk_percentage']
            )
            
            risk_amount = self.capital * risk_percentage

            # Calculate stop loss and take profit using hybrid ATR and predicted move approach
            atr_value = self.historical_data['ATR'].iloc[-1] if 'ATR' in self.historical_data else entry_price * 0.01
            
            # Calculate stop loss distance using hybrid approach
            atr_stop_distance = atr_value * self.trading_params['stop_loss_atr_multiplier']
            predicted_stop_distance = entry_price * predicted_move
            
            # Weighted combination of ATR and predicted move
            stop_loss_distance = (self.trading_params['atr_predicted_weight'] * atr_stop_distance + 
                                (1 - self.trading_params['atr_predicted_weight']) * predicted_stop_distance)

            # Calculate fee compensation factors
            P_in = current_price
            f_e = self.taker_fee
            f_tp = self.maker_fee
            f_sl = self.taker_fee
            
            # Calculate minimum stop loss distance to cover fees
            T_floor = P_in * (f_e + f_sl)
            
            if stop_loss_distance < T_floor:
                return

            P_sl = (P_in * (1 + f_e) - stop_loss_distance) / (1 - f_sl)
            T_rr = stop_loss_distance * self.trading_params['risk_reward_ratio']
            T = T_rr
            P_tp = (P_in * (1 + f_e) + T) / (1 - f_tp)
            P_be_tp = P_in * (1 + f_e) / (1 - f_tp)

            # Calculate quantity based on risk amount
            quantity = risk_amount / entry_price
            
            if quantity <= 0:
                return
            
            # Get trading filters and format quantities/prices
            try:
                tick_size, step_size, min_notional = get_filters(self.client, self.symbol)
                
                print(f"Symbol {self.symbol} trading filters:")
                print(f"  Tick size: {tick_size}")
                print(f"  Step size: {step_size}")
                print(f"  Min notional: {min_notional}")
                print(f"  Calculated quantity: {quantity}")
                
                # Format quantity according to LOT_SIZE step
                quantity_str = fmt_qty(quantity, step_size)
                quantity = float(quantity_str)
                
                # Format prices according to PRICE_FILTER tick
                entry_price_str = fmt_price(entry_price, tick_size)
                P_tp_str = fmt_price(P_tp, tick_size)
                P_sl_str = fmt_price(P_sl, tick_size)
                
                print(f"Formatted quantities and prices:")
                print(f"  Quantity: {quantity_str}")
                print(f"  Entry price: {entry_price_str}")
                print(f"  Take profit: {P_tp_str}")
                print(f"  Stop loss: {P_sl_str}")
                
                # Validate minimum notional
                notional_value = quantity * entry_price
                if min_notional > 0 and notional_value < min_notional:
                    print(f"❌ Notional value {notional_value} below minimum {min_notional}")
                    print(f"   Required minimum quantity: {min_notional / entry_price:.6f}")
                    return
                
                print(f"✅ Order validation passed - Notional: {notional_value}")
                
            except Exception as e:
                print(f"❌ Error getting trading filters: {e}")
                print("Proceeding with unformatted values...")
                # Fallback to basic validation
                if quantity <= 0:
                    return
                # Use unformatted values as fallback
                quantity_str = str(quantity)
                entry_price_str = str(entry_price)
                P_tp_str = str(P_tp)
                P_sl_str = str(P_sl)
                
            # Place the order using Binance API with OCO bracket
            try:
                # 1) Enter position with market buy
                buy_order = self.client.new_order(
                    symbol=self.symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=quantity_str
                )
                
                print(f"Market buy order placed: {buy_order}")

                executed = float(buy_order['executedQty'])
                
                # Calculate BNB commission paid (since BNB is used for fees)
                commission_bnb = sum(
                    float(o['commission']) for o in buy_order['fills']
                    if o['commissionAsset'] == 'BNB'
                )
                
                # Refresh account info to get current balances
                self.refresh_account_info()
                if not self.account_info:
                    print("❌ Failed to refresh account info")
                    return

                # Get current free balance
                free_asset = next(float(a['free']) for a in self.account_info['balances'] if a['asset'] == self.symbol_base)

                # Since BNB is used for fees, we can sell the full executed quantity
                sell_qty = floor_to_step(executed, step_size)
                
                # 2) Immediately attach OCO bracket order
                oco_order = self.client.new_oco_order(
                    symbol=self.symbol,
                    side='SELL',
                    quantity=sell_qty,

                    # ABOVE leg = take-profit limit (must be LIMIT_MAKER)
                    aboveType='LIMIT_MAKER',  # Type for take profit order
                    abovePrice=P_tp_str,  # Take profit price

                    # BELOW leg = stop (choose STOP_LOSS or STOP_LOSS_LIMIT)
                    belowType='STOP_LOSS',  # Type for stop loss order
                    belowStopPrice=P_sl_str,  # Stop loss trigger price

                    newOrderRespType='RESULT'  # Response type for the new order
                )
                
                print(f"OCO bracket order placed: {oco_order}")
                
                # Send success message
                print(f"✅ Trade successfully opened!")
                print(f"   Buy Order ID: {buy_order['orderId']}")
                print(f"   OCO Order List ID: {oco_order.get('orderListId', 'N/A')}")
                print(f"   Executed Quantity: {executed:.6f} {self.symbol_base}")
                print(f"   Sell Quantity: {sell_qty:.6f} {self.symbol_base}")
                print(f"   Commission Paid: {commission_bnb:.6f} BNB")
                
                # Store trade info for logging
                trade_info = {
                    'entry_price': entry_price,
                    'stop_loss': P_sl,
                    'take_profit': P_tp,
                    'quantity': quantity,
                    'entry_time': datetime.now(),
                    'buy_order': buy_order,
                    'oco_order': oco_order
                }
                
                self.trade_history.append(trade_info)
                
                print(f"Opened new trade with OCO bracket:")
                print(f"  Entry: ${entry_price:.4f}")
                print(f"  Stop Loss: ${P_sl:.4f}")
                print(f"  Take Profit: ${P_tp:.4f}")
                print(f"  Quantity: {quantity:.6f}")
                print(f"  Risk: {risk_percentage*100:.2f}%")

                # Calculate BNB fees paid (since BNB is used for fee structure)
                total_commission_bnb = sum(
                    float(fill['commission']) for fill in buy_order['fills']
                    if fill['commissionAsset'] == 'BNB'
                )
                
                # Calculate expected profit based on actual executed quantity
                expected_profit_usd = (float(P_tp_str) - float(entry_price_str)) * sell_qty
                expected_profit_pct = ((float(P_tp_str) - float(entry_price_str)) / float(entry_price_str)) * 100
                
                # Calculate expected loss if stop loss hits
                expected_loss_usd = (float(entry_price_str) - float(P_sl_str)) * sell_qty
                expected_loss_pct = ((float(entry_price_str) - float(P_sl_str)) / float(entry_price_str)) * 100

                send_telegram_message(f"""
                >> TRADE OPENED WITH OCO BRACKET <<
                Entry Time: {datetime.now()}

                📊 ORDER DETAILS:
                • Symbol: {self.symbol}
                • Side: BUY (Market) + SELL (OCO Bracket)

                💰 QUANTITIES & PRICES:
                • Expected Buy Qty: {quantity_str} {self.symbol_base}
                • Executed Buy Qty: {executed:.6f} {self.symbol_base}
                • Commission Paid: {commission_bnb:.6f} BNB
                • Sell Qty (OCO): {sell_qty:.6f} {self.symbol_base} (full executed qty - BNB fees)
                • Entry Price: ${entry_price_str}
                • Take Profit: ${P_tp_str}
                • Stop Loss: ${P_sl_str}

                💸 FEES & COSTS:
                • Commission Paid: {commission_bnb:.6f} BNB
                • Commission Asset: BNB (fee structure)

                📈 PROFIT/LOSS PROJECTIONS:
                • Expected Profit: ${expected_profit_usd:.2f} ({expected_profit_pct:.2f}%)
                • Expected Loss: ${expected_loss_usd:.2f} ({expected_loss_pct:.2f}%)
                • Risk/Reward Ratio: {expected_profit_usd/abs(expected_loss_usd):.2f}:1
                • Break Even Price: ${P_be_tp:.4f}

                🎯 TRADING PARAMETERS:
                • Risk Percentage: {risk_percentage*100:.2f}%
                • Predicted Move: {predicted_move:.6f}%
                • Order Type: OCO Bracket (LIMIT_MAKER + STOP_LOSS)
                • Order ID: {buy_order['orderId']}
                """)
                
                # Update capital from account after placing trade
                time.sleep(2)  # Wait for orders to process
                self.update_capital_from_account()
                
            except Exception as e:
                print(f"Error executing trade: {e}")
                # Cancel any orders that might have been placed
                try:
                    if 'buy_order' in locals():
                        self.client.cancel_order(symbol=self.symbol, orderId=buy_order['orderId'])
                    if 'oco_order' in locals():
                        # Cancel OCO order if it was created
                        self.client.cancel_oco_order(symbol=self.symbol, orderListId=oco_order['orderListId'])
                except:
                    pass
    
    def update_trades(self, current_price):
        """Update open trades and check for exits"""
        try:
            # Get current open orders from account
            open_orders = self.get_open_orders()
            if not open_orders:
                return
            
            print(f"Monitoring {len(open_orders)} open orders for {self.symbol}")
            
            # Check each order for potential manual exits
            for order in open_orders:
                order_id = order['orderId']
                order_type = order['type']
                side = order['side']
                
                # Only monitor sell orders (stop loss and take profit)
                if side == 'SELL':
                    # Check if this is a stop loss or take profit order
                    order_price = float(order['price'])
                    
                    # Handle OCO orders differently
                    if 'orderListId' in order:
                        # This is part of an OCO order
                        try:
                            # Get OCO order details to check creation time
                            oco_details = self.client.query_oco_order(symbol=self.symbol, orderListId=order['orderListId'])
                            created_time = datetime.fromtimestamp(oco_details['time'] / 1000)
                            holding_period = (datetime.now() - created_time).total_seconds() / 60
                            
                            if holding_period >= self.trading_params['max_holding_period']:
                                print(f"Cancelling old OCO order {order['orderListId']} (held for {holding_period:.1f} minutes)")
                                self.client.cancel_oco_order(symbol=self.symbol, orderListId=order['orderListId'])
                        except Exception as e:
                            print(f"Error checking OCO order {order['orderListId']}: {e}")
                    
                    # Handle regular stop orders
                    elif order_type == 'STOP_LOSS_LIMIT':
                        # This is likely a stop loss order
                        # Check if we should cancel it due to max holding period
                        try:
                            # Get order details to check creation time
                            order_details = self.client.query_order(symbol=self.symbol, orderId=order_id)
                            created_time = datetime.fromtimestamp(order_details['time'] / 1000)
                            holding_period = (datetime.now() - created_time).total_seconds() / 60
                            
                            if holding_period >= self.trading_params['max_holding_period']:
                                print(f"Cancelling old stop loss order {order_id} (held for {holding_period:.1f} minutes)")
                                self.client.cancel_order(symbol=self.symbol, orderId=order_id)
                        except Exception as e:
                            print(f"Error checking order {order_id}: {e}")
                            
        except Exception as e:
            print(f"Error updating trades: {e}")
    
    def close_trade(self, order_id, current_price, reason):
        """Close a trade by cancelling the order and placing a market sell"""
        try:
            # Get order details first to check if it's part of an OCO order
            order_details = self.client.query_order(symbol=self.symbol, orderId=order_id)
            quantity = float(order_details['origQty'])
            
            # Check if this is part of an OCO order
            if 'orderListId' in order_details and order_details['orderListId'] != -1:
                # Cancel the entire OCO order
                self.client.cancel_oco_order(symbol=self.symbol, orderListId=order_details['orderListId'])
                print(f"Cancelled OCO order {order_details['orderListId']}")
            else:
                # Cancel the individual order
                self.client.cancel_order(symbol=self.symbol, orderId=order_id)
                print(f"Cancelled order {order_id}")
            
            # Place market sell order
            sell_order = self.client.new_order(
                symbol=self.symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )
            
            print(f"Closed trade: Cancelled order {order_id}, placed market sell for {quantity}")
            
            # Update capital from account after trade closure
            time.sleep(2)  # Wait for order to process
            self.update_capital_from_account()
            
        except Exception as e:
            print(f"Error closing trade {order_id}: {e}")
    
    def load_historical_data_from_api(self):
        """
        Load X years of historical data from Binance API into memory.
        Maintains the same DataFrame structure as CSV loading for compatibility.
        """
        try:
            print(f"\n=== Loading {self.years_of_data} years of historical data from Binance API ===")
            
            # Calculate total bars needed (15-minute intervals)
            bars_per_year = 365 * 24 * 4
            total_bars_needed = int(self.years_of_data * bars_per_year)
            
            print(f"Target: {total_bars_needed:,} bars ({self.years_of_data} years of 15-minute data)")
            
            # Binance API limit is 1000 bars per request
            max_bars_per_request = 1000
            requests_needed = (total_bars_needed // max_bars_per_request) + 1
            
            print(f"Will need approximately {requests_needed} API requests")
            
            # No limit on requests - load all requested data
            print(f"Will make {requests_needed} API requests to load {total_bars_needed:,} bars")
            
            # Collect all data
            all_candles_data = []
            end_time = datetime.now(timezone.utc)
            
            print(f"Starting data collection from {end_time.strftime('%Y-%m-%d %H:%M:%S')} backwards...")
            print(f"Target symbol: {self.symbol}")
            
            consecutive_failures = 0
            max_consecutive_failures = 5
            
            for i in range(requests_needed):
                # Calculate time range for this request
                minutes_per_request = max_bars_per_request * 15
                current_start = end_time - timedelta(minutes=minutes_per_request)
                
                print(f"  Request {i+1:3d}/{requests_needed}: {current_start.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
                
                try:
                    klines = self.client.klines(
                        symbol=self.symbol,
                        interval='15m',
                        startTime=int(current_start.timestamp() * 1000),
                        endTime=int(end_time.timestamp() * 1000),
                        limit=1000
                    )
                    
                    if klines and len(klines) > 0:
                        all_candles_data.extend(klines)
                        consecutive_failures = 0  # Reset failure counter
                        print(f"    ✅ Got {len(klines)} bars (Total: {len(all_candles_data):,})")
                    else:
                        consecutive_failures += 1
                        print(f"    ❌ No data returned (consecutive failures: {consecutive_failures})")
                        
                    # Update end_time for next iteration
                    end_time = current_start
                    
                    # Rate limiting
                    time.sleep(0.1)  # 100ms between requests
                    
                    # Check if we have enough data
                    if len(all_candles_data) >= total_bars_needed:
                        print(f"    🎯 Reached target of {total_bars_needed:,} bars")
                        break
                    
                    # Stop if too many consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"    ⚠️  Stopping due to {consecutive_failures} consecutive failures")
                        break
                        
                except Exception as e:
                    consecutive_failures += 1
                    print(f"    ❌ Error in request {i+1}: {e} (consecutive failures: {consecutive_failures})")
                    
                    # Stop if too many consecutive failures
                    if consecutive_failures >= max_consecutive_failures:
                        print(f"    ⚠️  Stopping due to {consecutive_failures} consecutive failures")
                        break
                    
                    continue
            
            # Convert to DataFrame using the same method as existing code
            if all_candles_data:
                print(f"\n📊 Converting {len(all_candles_data):,} candles to DataFrame...")
                
                # Check if we have minimum required data
                min_required_bars = 1000  # At least 1000 bars (about 10 days)
                if len(all_candles_data) < min_required_bars:
                    print(f"⚠️  Warning: Only {len(all_candles_data)} bars collected (minimum recommended: {min_required_bars})")
                    print(f"   This may affect model training quality")
                
                # Use the existing conversion function
                self.raw_data = convert_binance_klines_to_dataframe(all_candles_data)
                
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
                    
                    # Check if we have recent data (within last 24 hours)
                    latest_date = self.historical_data['Date'].max()
                    # Ensure latest_date is timezone-aware
                    if latest_date.tzinfo is None:
                        latest_date = latest_date.replace(tzinfo=timezone.utc)
                    current_time = datetime.now(timezone.utc)
                    time_diff = (current_time - latest_date).total_seconds() / 3600  # hours
                    
                    print(f"   Latest data: {latest_date}")
                    print(f"   Current time: {current_time}")
                    print(f"   Time difference: {time_diff:.1f} hours")
                    
                    if time_diff > 24:
                        print(f"⚠️  Warning: Latest data is {time_diff:.1f} hours old")
                        print(f"   This might cause issues with WebSocket data processing")
                    
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
    
    def update_memory_with_new_candle(self, new_candle_df):
        """Update in-memory data with a new candle from WebSocket"""
        try:
            if self.raw_data.empty:
                print("❌ No existing raw data in memory")
                return False
            
            # Get the latest timestamp we already have
            latest_existing_date = self.raw_data['Date'].max()
            
            # Ensure latest_existing_date is timezone-aware
            if latest_existing_date.tzinfo is None:
                latest_existing_date = latest_existing_date.replace(tzinfo=timezone.utc)
            
            # Check if this is a new candle (after our latest timestamp)
            new_candle_date = new_candle_df['Date'].iloc[0]
            
            # Add more detailed debugging
            print(f"DEBUG: Latest in memory: {latest_existing_date}")
            print(f"DEBUG: New candle date: {new_candle_date}")
            print(f"DEBUG: Time difference: {(new_candle_date - latest_existing_date).total_seconds()} seconds")
            print(f"DEBUG: New candle is after latest: {new_candle_date > latest_existing_date}")
            
            # Check if this candle already exists (same timestamp)
            existing_candle = self.raw_data[self.raw_data['Date'] == new_candle_date]
            if not existing_candle.empty:
                print(f"DEBUG: Candle with timestamp {new_candle_date} already exists in memory")
                print(f"DEBUG: Existing candle OHLCV: O={existing_candle['Open'].iloc[0]:.4f}, H={existing_candle['High'].iloc[0]:.4f}, L={existing_candle['Low'].iloc[0]:.4f}, C={existing_candle['Close'].iloc[0]:.4f}")
                print(f"DEBUG: New candle OHLCV: O={new_candle_df['Open'].iloc[0]:.4f}, H={new_candle_df['High'].iloc[0]:.4f}, L={new_candle_df['Low'].iloc[0]:.4f}, C={new_candle_df['Close'].iloc[0]:.4f}")
                
                # Check if the data is different (candle was updated)
                data_changed = (existing_candle['Open'].iloc[0] != new_candle_df['Open'].iloc[0] or
                               existing_candle['High'].iloc[0] != new_candle_df['High'].iloc[0] or
                               existing_candle['Low'].iloc[0] != new_candle_df['Low'].iloc[0] or
                               existing_candle['Close'].iloc[0] != new_candle_df['Close'].iloc[0])
                
                if data_changed:
                    print(f"DEBUG: Candle data has changed, updating existing candle")
                    # Update the existing candle
                    self.raw_data.loc[self.raw_data['Date'] == new_candle_date, ['Open', 'High', 'Low', 'Close', 'Volume']] = [
                        new_candle_df['Open'].iloc[0],
                        new_candle_df['High'].iloc[0], 
                        new_candle_df['Low'].iloc[0],
                        new_candle_df['Close'].iloc[0],
                        new_candle_df['Volume'].iloc[0]
                    ]
                    
                    # Recalculate technical indicators
                    print(f"Recalculating technical indicators with updated candle...")
                    data_for_indicators = self.raw_data.copy()
                    if 'Date' in data_for_indicators.columns:
                        data_for_indicators = data_for_indicators.set_index('Date')
                    
                    self.historical_data = calculate_technical_indicators(data_for_indicators, None)
                    
                    # Clean up Date ambiguity
                    if 'Date' in self.historical_data.columns and hasattr(self.historical_data.index, 'name') and self.historical_data.index.name == 'Date':
                        self.historical_data = self.historical_data.drop(columns=['Date']).reset_index()
                    elif hasattr(self.historical_data.index, 'name') and self.historical_data.index.name == 'Date':
                        self.historical_data = self.historical_data.reset_index()
                    elif 'Date' not in self.historical_data.columns:
                        self.historical_data = self.historical_data.reset_index()
                        if 'index' in self.historical_data.columns:
                            self.historical_data = self.historical_data.rename(columns={'index': 'Date'})
                    
                    print(f"✅ Updated existing candle in memory")
                    return True
                else:
                    print(f"DEBUG: Candle data is identical, no update needed")
                    return True
            
            # If we get here, it's a truly new candle
            if new_candle_date <= latest_existing_date:
                print(f"DEBUG: No new candle found. Latest in memory: {latest_existing_date}, New candle: {new_candle_date}")
                print(f"DEBUG: This candle is not newer than our latest data")
                return True
            
            print(f"DEBUG: Found truly new candle to add to memory: {new_candle_date}")
            print(f"DEBUG: This is a new 15-minute candle that we haven't seen before")
            
            # Add new candle to raw data
            self.raw_data = pd.concat([self.raw_data, new_candle_df], ignore_index=True)
            self.raw_data = self.raw_data.sort_values('Date').reset_index(drop=True)
            
            # Recalculate technical indicators for the full updated dataset
            print(f"Recalculating technical indicators with new candle...")
            
            # Prepare data for technical indicators (expects Date as index)
            data_for_indicators = self.raw_data.copy()
            if 'Date' in data_for_indicators.columns:
                data_for_indicators = data_for_indicators.set_index('Date')
            
            self.historical_data = calculate_technical_indicators(data_for_indicators, None)
            
            # Clean up Date ambiguity: ensure Date is ONLY a column, not index
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
            
            # Update running statistics with the new candle
            if self.current_features and self.running_stats:
                print(f"Updating running statistics with new candle...")
                
                # Get the technical indicators for this candle from the updated historical data
                candle_date = new_candle_date
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
            
            # Initialize PSI tracking variables if not set
            if self.last_retrain_bar is None:
                self.last_retrain_bar = len(self.raw_data) - 1
            if self.segment_start is None:
                self.segment_start = len(self.raw_data) - 1
            
            latest_close = new_candle_df['Close'].iloc[-1]
            print(f"✅ Added new candle to memory")
            print(f"   Latest: {self.last_processed_timestamp} - Close: ${latest_close:.4f}")
            print(f"   Total bars in memory: {len(self.raw_data):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating memory with new candle: {e}")
            return False
    
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
        df_processed, feature_cols, target_cols = preprocess_data(df)
        
        # Calculate initial feature importance
        self.current_features = calculate_feature_importance(
            df_processed,
            feature_cols,
            target_cols,
            model_params=self.model_params,
            iterations=1,
            save_importance=False,
            visualize_importance=False,
            K=self.trading_params['feature_count_k']
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
        training_start_idx = max(0, len(df_processed) - self.window_size)
        training_df = df_processed.iloc[training_start_idx:].copy()
        
        print(f"Training on most recent {len(training_df)} bars (window_size: {self.window_size})")
        print(f"Full context: {len(df_processed)} bars (buffer provided proper indicator calculation)")
        
        # Process each row sequentially to build training data with incremental standardization
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
            
            # Initialize PSI reference data from the training window
            print("Initializing PSI reference data...")
            training_window_data = df_processed.iloc[training_start_idx:].copy()
            self.psi_ref_data = training_window_data[self.current_features]
            self.psi_ref_bins = {}
            for feature in self.current_features:
                if feature in self.psi_ref_data.columns:
                    self.psi_ref_bins[feature] = make_ref_bins(
                        self.psi_ref_data[feature].values, 
                        bins=self.psi_params['psi_bins']
                    )
            
            # Initialize PSI tracking variables
            self.last_retrain_bar = len(df_processed) - 1
            self.segment_start = len(df_processed) - 1
            
            print(f"PSI reference data initialized with {len(self.psi_ref_data)} samples")
            
            # Log training sanity check
            self.sanity_log(self.model, X_initial[-1:], tag="INIT_TRAIN")
        print("Model initialized successfully")
    
    def update_model(self, df=None):
        """Update the model with new data based on PSI drift detection"""
        # Use the full in-memory historical data (already has technical indicators)
        if df is None:
            if self.historical_data.empty:
                print("No historical data available for model retraining")
                return
            df = self.historical_data.copy()
        
        # Check for PSI drift
        current_data_length = len(df)
        if not self.check_psi_drift(current_data_length):
            return
        
        print(f"\nRetraining model at {datetime.now().strftime('%H:%M:%S')} due to PSI drift detection...")
        
        if current_data_length < self.window_size:
            print(f"Insufficient data for model retraining: {current_data_length} < {self.window_size}")
            return
            
        print(f"Using {current_data_length:,} bars from full historical dataset for retraining")
        
        # Data already has technical indicators calculated, just preprocess
        df_processed, feature_cols, target_cols = preprocess_data(df)
        
        # Recalculate feature importance
        print("Recalculating feature importance...")
        self.current_features = calculate_feature_importance(
            df_processed,
            feature_cols,
            target_cols,
            model_params=self.model_params,
            iterations=1,
            save_importance=False,
            visualize_importance=False,
            K=self.trading_params['feature_count_k']
        )
        
        # Get preprocessed features and target using incremental standardization
        X = []
        y = []
        
        # Use current running stats for retraining
        temp_running_stats = {col: self.running_stats[col].copy() for col in self.current_features if col in self.running_stats}
        
        # Use only the most recent window_size bars for retraining (but with full context for indicators)
        training_start_idx = max(0, len(df_processed) - self.window_size)
        training_df = df_processed.iloc[training_start_idx:].copy()
        
        print(f"Retraining on most recent {len(training_df)} bars (window_size: {self.window_size})")
        print(f"Full context: {len(df_processed)} bars (buffer provided proper indicator calculation)")
        
        # Process each row sequentially to build training data with incremental standardization
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
            
            # Update PSI reference data with new training window
            print("Updating PSI reference data...")
            self.psi_ref_data = training_df[self.current_features]
            self.psi_ref_bins = {}
            for feature in self.current_features:
                if feature in self.psi_ref_data.columns:
                    self.psi_ref_bins[feature] = make_ref_bins(
                        self.psi_ref_data[feature].values, 
                        bins=self.psi_params['psi_bins']
                    )
            
            # Update PSI tracking variables
            self.last_retrain_bar = current_data_length - 1
            self.segment_start = current_data_length - 1
            
            print(f"PSI reference data updated with {len(self.psi_ref_data)} samples")
            
            # Log retraining sanity check
            self.sanity_log(self.model, X[-1:], tag="PSI_RETRAIN")
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
    
    def check_psi_drift(self, current_data_length):
        """Check for PSI drift and determine if model should be retrained"""
        if self.psi_ref_data is None or len(self.psi_ref_bins) == 0:
            return False
        
        # Check if we have enough data for PSI calculation
        if current_data_length < self.psi_params['psi_window']:
            return False
        
        # Check if enough time has passed since last retrain
        if self.last_retrain_bar is not None:
            bars_since_last_retrain = current_data_length - self.last_retrain_bar
            if bars_since_last_retrain < self.psi_params['psi_cooldown']:
                return False
        
        # Check if we have a minimum segment size
        if self.segment_start is not None:
            current_segment_size = current_data_length - self.segment_start
            if current_segment_size < self.psi_params['min_segment']:
                return False
        
        # Check if it's time for a PSI check (every psi_check_step bars)
        if self.segment_start is not None:
            bars_since_segment_start = current_data_length - self.segment_start
            if bars_since_segment_start % self.psi_params['psi_check_step'] != 0:
                return False
        
        # Calculate PSI for current window
        try:
            # Get current window data
            current_window_start = current_data_length - self.psi_params['psi_window']
            current_window_data = self.historical_data.iloc[current_window_start:current_data_length]
            
            # Preprocess current window
            current_processed, _, _ = preprocess_data(current_window_data)
            
            # Calculate PSI for each feature
            max_psi = 0.0
            for feature in self.current_features:
                if feature in self.psi_ref_bins and feature in current_processed.columns:
                    psi_value = psi_from_bins(
                        self.psi_ref_data[feature].values,
                        current_processed[feature].values,
                        self.psi_ref_bins[feature]
                    )
                    max_psi = max(max_psi, psi_value)
            
            print(f"PSI check: max_psi = {max_psi:.4f}")
            
            # Check if drift exceeds threshold
            if max_psi > self.psi_params['psi_threshold_hi']:
                print(f"Strong drift detected (PSI={max_psi:.4f} > {self.psi_params['psi_threshold_hi']})")
                return True
            elif max_psi > self.psi_params['psi_threshold_lo']:
                print(f"Mild drift detected (PSI={max_psi:.4f} > {self.psi_params['psi_threshold_lo']})")
                # Continue monitoring but note mild drift
            
            return False
            
        except Exception as e:
            print(f"Error in PSI drift check: {e}")
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
        """Log model and prediction sanity checks to a file"""
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
                values = X_row.values
                X_for_apply = X_row.values
            elif isinstance(X_row, np.ndarray):
                values = X_row
                X_for_apply = X_row
            else:
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
        """Log leaf index diversity for the last Kb rows to detect model stagnation"""
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
    
    def start_live_trading(self):
        """Start live trading with WebSocket kline streams"""
        if not self.client:
            print("No authenticated client available")
            return
            
        print(f"\nStarting live trading for {self.symbol}...")
        print(f"Trading parameters: {self.trading_params}")
        
        # Sync with account orders first
        print("\nSyncing with account orders...")
        sync_result = self.sync_with_account_orders()
        print(f"sync_with_account_orders() returned: {sync_result}")
        
        # Initialize model (uses full in-memory dataset)
        self.initialize_model()
        
        # Make initial prediction on current bar before starting WebSocket stream
        print("\nMaking initial prediction on current bar...")
        initial_prediction = self.predict_next_bar()
        if initial_prediction is not None:
            current_price = self.raw_data['Close'].iloc[-1]
            print(f"Initial prediction: {initial_prediction:.4f} at price ${current_price:.4f}")
            
            # Execute trade if conditions are met
            self.execute_trade(initial_prediction, current_price)
            
            # Print initial trading summary
            open_trades_count = self.get_open_trades_count()
            print(f"Initial state - Capital: ${self.capital:,.2f}, Open trades: {open_trades_count}")
        else:
            print("No initial prediction available")
        
        print("\nLive trading started. Press Ctrl+C to stop.")
        print(f"Data updates: Real-time via WebSocket kline streams")
        print(f"Model retraining: PSI-based drift detection")
        print(f"PSI parameters: check_step={self.psi_params['psi_check_step']}, threshold_hi={self.psi_params['psi_threshold_hi']}")
        
        try:
            # Start the kline stream
            self.start_kline_stream()
            
            # Keep the main thread alive and monitor WebSocket health
            last_heartbeat = datetime.now()
            heartbeat_interval = 60  # Print heartbeat every 30 seconds
            
            while self.stream_active:
                current_time = datetime.now()
                
                # Print heartbeat to show the bot is still running
                if (current_time - last_heartbeat).total_seconds() >= heartbeat_interval:
                    print(f"[HEARTBEAT] Bot is running... Stream active: {self.stream_active}")
                    print(f"[HEARTBEAT] Current time: {current_time.strftime('%H:%M:%S')}")
                    print(f"[HEARTBEAT] Latest data: {self.raw_data['Date'].max() if not self.raw_data.empty else 'No data'}")
                    last_heartbeat = current_time
                
                time.sleep(1)
                    
        except KeyboardInterrupt:
            print("\nStopping live trading...")
            self.stop_kline_stream()
            print(f"Final capital: ${self.capital:,.2f}")
            print(f"Total trades: {len(self.trade_history)}")
            if self.trade_history:
                total_profit = sum(trade.get('profit', 0) for trade in self.trade_history)
                print(f"Total P&L: ${total_profit:.2f}")


def convert_binance_klines_to_dataframe(klines_data):
    """
    Convert Binance kline data to DataFrame format expected by calculate_technical_indicators
    
    Args:
        klines_data: List of kline arrays from Binance API
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    if not klines_data:
        return pd.DataFrame()
    
    # Convert the data to a list of dictionaries
    data_list = []
    
    for kline in klines_data:
        # Extract data from kline array
        data_list.append({
            'Open': float(kline[1]),
            'High': float(kline[2]),
            'Low': float(kline[3]),
            'Close': float(kline[4]),
            'Volume': float(kline[5])
        })
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Create datetime index from timestamps
    timestamps = [kline[0] for kline in klines_data]
    df.index = pd.to_datetime(timestamps, unit='ms', utc=True)  # Convert from Unix timestamp with UTC timezone
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_index()
    
    print(f"Converted {len(df)} data points from Binance API")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def load_binance_api_key():
    """Load API credentials from Binance JSON file"""
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'settings', 'binance_api_key.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract API key and secret
        api_key = data.get('api_key')
        api_secret = data.get('api_secret')
        
        if not api_key or not api_secret:
            print("Error: Missing 'api_key' or 'api_secret' in Binance API key file")
            return None, None
        
        return api_key, api_secret
        
    except FileNotFoundError:
        print(f"Error: Binance API key file not found at {json_path}")
        return None, None
    except json.JSONDecodeError:
        print("Error: Invalid JSON in Binance API key file")
        return None, None
    except Exception as e:
        print(f"Error loading Binance API key: {e}")
        return None, None


def main():
    print("=== Binance API Data Fetching with WebSocket Streams ===")
    
    # Load API credentials from Binance JSON file
    print("\nLoading API credentials...")
    api_key, api_secret = load_binance_api_key()
    
    if not api_key or not api_secret:
        print("Failed to load API credentials. Exiting.")
        return
    
    # Initialize Binance API client with 2 years of historical data
    print("Initializing API client with historical data loading...")
    years_data = 6  # Default to 2 years
    try:
        years_data = float(years_data) if years_data else 2.0
    except ValueError:
        years_data = 2.0
    
    api = BinanceAPIClient(api_key=api_key, api_secret=api_secret, years_of_data=years_data)
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
    account_info = api.get_account_info()
    if account_info:
        # Look for USDC balance
        for balance in account_info['balances']:
            if balance['asset'] == 'USDC':
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_balance = free_balance + locked_balance
                print(f"USDC Balance: Free={free_balance:.2f}, Locked={locked_balance:.2f}, Total={total_balance:.2f}")
                api.capital = total_balance
                break
    else:
        print("  Failed to fetch accounts")
    
    # 4. Fetch current orders
    print("\n4. Fetching current orders...")
    open_orders = api.get_open_orders()
    if open_orders:
        print(f"Found {len(open_orders)} open orders")
        for order in open_orders[:3]:  # Show first 3 orders
            print(f"  - {order['symbol']}: {order['side']} {order['origQty']} @ {order['price']} ({order['type']})")
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
    
    # Start live trading immediately
    print(f"\n{'='*50}")
    print("🚀 STARTING LIVE TRADING WITH WEBSOCKET STREAMS...")
    print(f"{'='*50}")
    api.start_live_trading()


if __name__ == "__main__":
    main()
