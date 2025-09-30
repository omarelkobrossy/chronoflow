import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
import optuna
import json
import math
from utils import preprocess_data, calculate_feature_importance, clamp, calculate_sharpe_ratio, calculate_max_drawdown, calculate_distribution_metrics, calculate_cagr, calculate_mar, calculate_composite_score, sanitize_features, standardize_expanding, standardize_expanding_train
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from Optimization.FAPT_Wasserstein import predict_optimal_parameters, get_top_market_weather_features
import cupy as cp


symbol = "XRP_USD"

# Create T and H dictionaries for default parameters
T_default = {
    "min_risk_percentage": 0.18667545873693023,
    "max_risk_percentage": 0.7333546909470524,
    "risk_scaling_factor": 2.5184557369439187,
    "risk_reward_ratio": 1.5264918456689758,
    "min_predicted_move": 0.007964038055791422,
    "partial_take_profit": 0.7091585844137864,
    "min_holding_period": 14,
    "max_holding_period": 81,
    "max_concurrent_trades": 8,
    "stop_loss_atr_multiplier": 3.9267721668435853,
    "atr_predicted_weight": 0.6210460131734021,
    "aggressiveness": 1.933354082966644,
    "feature_count_k": 35,
    "window_size": 34257,
    "retrain_interval": 30333
}

H_default = {
    "learning_rate": 0.04124763223591977,
    "n_estimators": 1268,
    "max_depth": 5,
    "max_leaves": 51,
    "min_child_weight": 0.7487494959921415,
    "gamma": 0.217123826259615,
    "subsample": 0.6064938340420224,
    "colsample_bytree": 0.5213054854581245,
    "colsample_bylevel": 0.7647516429830787,
    "reg_lambda": 1.9815198277316892,
    "reg_alpha": 6.133298215314081e-05,
    "max_bin": 874,
    'random_state': 42,
    'tree_method': 'hist',
    'device': 'cpu',
}

# Fixed parameters
INITIAL_CAPITAL = 2000
SLIPPAGE = 0.000
MAKER_FEE = 0.006  # 0.6% fee when buying (adding to trade value)
TAKER_FEE = 0.012  # 1.2% fee when selling (deducted from sale value)


# Flag to skip optimization
SKIP_OPTIMIZATION = True  # Set to True to use default parameters
USE_FAPT = False
OPTIMIZATION_TRIALS = 4000
RESUME_STUDY = True  # Set to True to resume from previous study, False to start new, None to check if exists

MIN_WINDOW = 300
MAX_WINDOW = 50000
EQUITY_CURVE_PHASES = 4

if USE_FAPT:

    SKIP_OPTIMIZATION = True
    global_market_weather_features = get_top_market_weather_features(symbol)
    market_weather_features = {}
    for feature in global_market_weather_features:
        base_feature = "_".join(feature.split('_')[:-1])  # Get the base feature name without the metric suffix
        if base_feature not in market_weather_features:
            market_weather_features[base_feature] = []
        stat_type = feature.split('_')[-1]  # Get the metric suffix (e.g., 'mean', 'std', etc.)
        market_weather_features[base_feature].append(stat_type)


def run_strategy(df_window, T, H, feature_cols, target_cols):
 
    # Initialize predicted change column
    df_window['Predicted_Change'] = np.nan
    
    # Simulate trading with model retraining every retrain_interval bars
    capital = INITIAL_CAPITAL
    open_trades = []
    trade_history = []
    holdout_predictions = []
    holdout_actuals = []
    rolling_metrics = []
    
    # Enhanced Cash Ledger System
    class CashLedger:
        def __init__(self, initial_capital):
            self.initial_capital = initial_capital
            self.available_cash = initial_capital
            self.invested_cash = 0.0  # Cash tied up in open trades
            self.total_fees_paid = 0.0  # Track total fees paid
            self.total_profit_loss = 0.0  # Track total P&L
            self.trade_count = 0
            
        def open_trade(self, trade_value, taker_fee):
            """Open a new trade and update cash ledger"""
            if trade_value > self.available_cash:
                raise ValueError(f"Insufficient cash: need ${trade_value:.2f}, have ${self.available_cash:.2f}")
            
            self.available_cash -= trade_value
            self.invested_cash += trade_value
            self.total_fees_paid += taker_fee
            self.trade_count += 1
            
        def close_trade(self, trade_value, profit_loss, maker_fee):
            """Close a trade and update cash ledger"""
            # Recover the original invested cash
            self.available_cash += trade_value
            self.invested_cash -= trade_value
            
            # Add the profit/loss
            self.available_cash += profit_loss
            self.total_profit_loss += profit_loss
            self.total_fees_paid += maker_fee
            
        def get_total_capital(self):
            """Get total capital (initial + P&L)"""
            return self.initial_capital + self.total_profit_loss
            
        def get_portfolio_value(self, open_trades_market_value):
            """Get total portfolio value (available cash + invested cash + unrealized P&L)"""
            return self.available_cash + self.invested_cash + open_trades_market_value
            
        def get_status(self):
            """Get current cash ledger status"""
            return {
                'initial_capital': self.initial_capital,
                'available_cash': self.available_cash,
                'invested_cash': self.invested_cash,
                'total_capital': self.get_total_capital(),
                'total_fees_paid': self.total_fees_paid,
                'total_profit_loss': self.total_profit_loss,
                'trade_count': self.trade_count
            }
    
    cash_ledger = CashLedger(INITIAL_CAPITAL)
    
    # Initialize equity curve only for the trading period (after initial training window)
    trading_start_idx = T['window_size']
    equity_curve = pd.Series(index=df_window['Date'].iloc[trading_start_idx:], data=np.nan, dtype=np.float64)
    
    n = len(df_window)
    
    # =========================
    # Initialize model & FI
    # =========================
    print("\nInitializing first window...")
    initial_window = df_window.iloc[:T['window_size']].copy()

    # 1) Feature importance on the initial window
    current_features = calculate_feature_importance(
        initial_window,
        feature_cols,
        target_cols,
        model_params=H,
        iterations=1,
        save_importance=False,
        visualize_importance=False,
        K=T['feature_count_k']
    )

    # 2) Sanitize the selected features for THIS training slice
    clean_features = sanitize_features(initial_window, current_features)

    # If too few survived, fall back to generic feature_cols
    if len(clean_features) < 3:
        fallback = sanitize_features(initial_window, feature_cols)
        clean_features = fallback[:max(3, min(10, len(fallback)))]

    # 3) Standardize expanding (no lookahead) for the training window
    X_initial = standardize_expanding_train(initial_window, clean_features)

    # 4) Target aligned to the same rows
    y_initial = df_window.loc[X_initial.index, target_cols].values.ravel().astype(np.float32)

    # 5) Build & fit model (ensure GPU params are unified everywhere)
    H['device'] = 'cuda'
    H['max_bin'] = 512   # pick one value and keep it the same across all trainings

    model = xgb.XGBRegressor(**H)
    model.fit(X_initial, y_initial)
    
    # Process data in chunks
    print("\nProcessing data in chunks...")
    total_data_processed = 0
    
    for start in range(T['window_size'], n, T['retrain_interval']):
        end = min(start + T['retrain_interval'], n)
        total_data_processed += T['retrain_interval']
        
        # Recalculate feature importance every window_size worth of data
        if total_data_processed >= T['retrain_interval']:
            print(f"\nRecalculating feature importance at index {start}...")
            feature_selection_data = df_window.iloc[start-T['window_size']:start].copy()
            current_features = calculate_feature_importance(
                feature_selection_data,
                feature_cols,
                target_cols,
                model_params=H,
                iterations=1,
                save_importance=False,
                visualize_importance=False,
                K=T['feature_count_k']
            )
            total_data_processed = 0
        
        current_holdout = df_window.iloc[start:end].copy()
        current_train   = df_window.iloc[:start].copy()

        # 1) sanitize feature list for THIS train slice
        clean_features = sanitize_features(current_train, current_features)

        # fail early if too few features survived
        if len(clean_features) < 3:
            # fallback: keep top-3 present numeric features from feature_cols
            fallback = sanitize_features(current_train, feature_cols)
            clean_features = fallback[:max(3, min(10, len(fallback)))]

        # 2) standardize with identical order & dtype
        Xtr, Xho = standardize_expanding(current_train, current_holdout, clean_features)
        ytr = current_train[target_cols].values.ravel().astype(np.float32)
        yho = current_holdout[target_cols].values.ravel().astype(np.float32)

        # 3) train
        model = xgb.XGBRegressor(**H)
        model.fit(Xtr, ytr)

        # 4) lock the exact order booster expects; reindex holdout defensively
        expected_order = model.get_booster().feature_names
        # (When fitting with pandas, xgboost keeps column names in order.)
        # Reindex just in case
        Xho = Xho.reindex(columns=expected_order)
        Xho_cu = cp.asarray(Xho.values, dtype=np.float32)

        # Optional: assert to catch mismatches immediately
        assert list(Xtr.columns) == expected_order, "Train column order mismatch"
        assert list(Xho.columns) == expected_order, "Holdout column order mismatch"

        # 5) predict
        booster = model.get_booster()
        preds_cu = booster.inplace_predict(Xho_cu)
        chunk_predictions = cp.asnumpy(preds_cu).astype(np.float32, copy=False)

        # write back
        df_window.loc[df_window.index[start:end], 'Predicted_Change'] = chunk_predictions
        holdout_predictions.extend(chunk_predictions)
        holdout_actuals.extend(yho)

        # Calculate metrics for this chunk
        chunk_mse = mean_squared_error(yho, chunk_predictions)
        chunk_r2 = r2_score(yho, chunk_predictions)
        chunk_dir_acc = np.mean((yho > 0) == (chunk_predictions > 0))
        
        rolling_metrics.append({
            'Start_Date': current_holdout.index[0],
            'End_Date': current_holdout.index[-1],
            'MSE': chunk_mse * 100,
            'R2': chunk_r2,
            'Directional_Accuracy': chunk_dir_acc,
            'Training_Size': len(current_train)
        })
        
        # Get current market weather metrics for FAPT if enabled
        if USE_FAPT:
            current_feature_metrics = {}
            for feature, stat_types in market_weather_features.items():
                feature_series = pd.Series(current_holdout[feature].values)
                if feature_series.nunique() <= 2:  # Skip binary or constant features
                   continue
                
                # Get all metrics for this feature using calculate_distribution_metrics
                metrics = calculate_distribution_metrics(feature_series)
                if metrics is not None:
                    # Only keep the metrics that are in our stat_types list
                    for stat_type in stat_types:
                        if stat_type in metrics:
                            current_feature_metrics[f"{feature}_{stat_type}"] = metrics[stat_type]
            
            # Get optimal parameters from FAPT
            fapt_params = predict_optimal_parameters(current_feature_metrics, symbol)
            min_risk_percentage = fapt_params['parameters']['min_risk_percentage']
            max_risk_percentage = fapt_params['parameters']['max_risk_percentage']
            risk_scaling_factor = fapt_params['parameters']['risk_scaling_factor']
            risk_reward_ratio = fapt_params['parameters']['risk_reward_ratio']
            min_predicted_move = fapt_params['parameters']['min_predicted_move']
            partial_take_profit = fapt_params['parameters']['partial_take_profit']
            min_holding_period = fapt_params['parameters']['min_holding_period']
            max_holding_period = fapt_params['parameters']['max_holding_period']
            max_concurrent_trades = fapt_params['parameters']['max_concurrent_trades']

        # Run trading for this window
        for idx in range(start, end):
            row = df_window.iloc[idx]
            
            # Update open trades
            closed_trades = []
            for trade in open_trades:
                high = row['High']
                low = row['Low']
                holding_period = idx - trade['entry_idx']
                
                if high >= trade['take_profit']:
                    exit_price = trade['take_profit'] * (1 - SLIPPAGE)
                    # Calculate profit with Maker fee (0.6% deducted from sale value)
                    gross_profit = (exit_price - trade['entry_price']) * trade['size']
                    sale_value = exit_price * trade['size']
                    maker_fee_amount = sale_value * MAKER_FEE
                    profit = gross_profit - maker_fee_amount
                    
                    # Update cash ledger
                    cash_ledger.close_trade(trade['trade_value'], profit, maker_fee_amount)
                    
                    trade['exit_idx'] = idx
                    trade['exit_price'] = exit_price
                    trade['result'] = 'TP'
                    trade['profit'] = profit
                    trade_history.append(trade)
                    closed_trades.append(trade)
                elif low <= trade['stop_loss']:
                    exit_price = trade['stop_loss'] * (1 - SLIPPAGE)
                    # Calculate profit with Maker fee (0.6% deducted from sale value)
                    gross_profit = (exit_price - trade['entry_price']) * trade['size']
                    sale_value = exit_price * trade['size']
                    maker_fee_amount = sale_value * MAKER_FEE
                    profit = gross_profit - maker_fee_amount
                    
                    # Update cash ledger
                    cash_ledger.close_trade(trade['trade_value'], profit, maker_fee_amount)
                    
                    trade['exit_idx'] = idx
                    trade['exit_price'] = exit_price
                    trade['result'] = 'SL'
                    trade['profit'] = profit
                    trade_history.append(trade)
                    closed_trades.append(trade)
                elif holding_period >= T['min_holding_period']:
                    projected_tp = trade['take_profit']
                    projected_entry = trade['entry_price']
                    # Calculate partial take profit with fee compensation
                    # The partial TP should be adjusted to account for the maker fee
                    maker_fee_factor = 1 - MAKER_FEE
                    tp_partial = projected_entry + T['partial_take_profit'] * (projected_tp - projected_entry) / maker_fee_factor
                    if high >= tp_partial:
                        exit_price = tp_partial * (1 - SLIPPAGE)
                        # Calculate profit with Maker fee (0.6% deducted from sale value)
                        gross_profit = (exit_price - trade['entry_price']) * trade['size']
                        sale_value = exit_price * trade['size']
                        maker_fee_amount = sale_value * MAKER_FEE
                        profit = gross_profit - maker_fee_amount
                        
                        # Update cash ledger
                        cash_ledger.close_trade(trade['trade_value'], profit, maker_fee_amount)
                        
                        trade['exit_idx'] = idx
                        trade['exit_price'] = exit_price
                        trade['result'] = 'Partial TP'
                        trade['profit'] = profit
                        trade_history.append(trade)
                        closed_trades.append(trade)
                    elif low <= projected_entry:
                        exit_price = projected_entry * (1 - SLIPPAGE)
                        # Calculate profit with Maker fee (0.6% deducted from sale value)
                        gross_profit = (exit_price - trade['entry_price']) * trade['size']
                        sale_value = exit_price * trade['size']
                        maker_fee_amount = sale_value * MAKER_FEE
                        profit = gross_profit - maker_fee_amount
                        
                        # Update cash ledger
                        cash_ledger.close_trade(trade['trade_value'], profit, maker_fee_amount)
                        
                        trade['exit_idx'] = idx
                        trade['exit_price'] = exit_price
                        trade['result'] = 'BE'
                        trade['profit'] = profit
                        trade_history.append(trade)
                        closed_trades.append(trade)
                
                if holding_period >= T['max_holding_period'] and trade not in closed_trades:
                    exit_price = row['Close'] * (1 - SLIPPAGE)
                    # Calculate profit with Maker fee (0.6% deducted from sale value)
                    gross_profit = (exit_price - trade['entry_price']) * trade['size']
                    sale_value = exit_price * trade['size']
                    maker_fee_amount = sale_value * MAKER_FEE
                    profit = gross_profit - maker_fee_amount
                    
                    # Update cash ledger
                    cash_ledger.close_trade(trade['trade_value'], profit, maker_fee_amount)
                    
                    trade['exit_idx'] = idx
                    trade['exit_price'] = exit_price
                    trade['result'] = 'MAXHOLD'
                    trade['profit'] = profit
                    trade_history.append(trade)
                    closed_trades.append(trade)
            
            open_trades = [t for t in open_trades if t not in closed_trades]
            
            # Entry logic
            if len(open_trades) < T['max_concurrent_trades']:
                if row['Predicted_Change'] < -T['min_predicted_move']:
                    entry_price = row['Open'] * (1 + SLIPPAGE)
                    
                    predicted_move = abs(row['Predicted_Change'])
                    
                    # Calculate how much the predicted move exceeds the minimum threshold
                    move_excess_ratio = (predicted_move - T['min_predicted_move']) / T['min_predicted_move']
                    
                    # Apply aggressiveness scaling: higher aggressiveness = faster scaling to max risk
                    # aggressiveness=1: linear scaling, aggressiveness=2: quadratic scaling, etc.
                    scaled_excess = move_excess_ratio ** (1 / T['aggressiveness'])
                    
                    # Calculate risk percentage with aggressiveness-controlled scaling
                    risk_percentage = min(
                        T['min_risk_percentage'] + (T['max_risk_percentage'] - T['min_risk_percentage']) * min(scaled_excess * T['risk_scaling_factor'], 1.0),
                        T['max_risk_percentage']
                    )
                    
                    # Use the cash ledger to get accurate available cash
                    available_cash = cash_ledger.available_cash
                    #print(f"available_cash: {available_cash}")
                    
                    # Only proceed if we have enough cash for this trade
                    if available_cash <= 0:
                        continue
                    
                    # Debug: Show cash management for first few trades
                    # if len(trade_history) < 5:
                    #     status = cash_ledger.get_status()
                    #     print(f"  Trade #{len(trade_history)+1}: Total Capital=${status['total_capital']:.2f}, Available Cash=${available_cash:.2f}, Invested=${status['invested_cash']:.2f}")
                    
                    # Use available cash for risk calculation
                    risk_amount = available_cash * risk_percentage
                    # Calculate stop loss and take profit using hybrid ATR and predicted move approach
                    atr_value = row['ATR'] if 'ATR' in row else row['Close'] * 0.01  # Fallback to 1% if ATR not available
                    
                    # Calculate stop loss distance using hybrid approach
                    atr_stop_distance = atr_value * T['stop_loss_atr_multiplier']
                    predicted_stop_distance = entry_price * predicted_move  # Convert predicted move to price distance
                    
                    # Weighted combination of ATR and predicted move
                    stop_loss_distance = (T['atr_predicted_weight'] * atr_stop_distance + 
                                        (1 - T['atr_predicted_weight']) * predicted_stop_distance)
                    
                    # Calculate fee compensation factors
                    P_in = entry_price
                    f_e = TAKER_FEE
                    f_tp = MAKER_FEE
                    f_sl = TAKER_FEE

                    # Calculate minimum stop loss distance to cover fees
                    # T_floor = P_in * (f_e + f_sl) ensures we don't lose money on fees
                    T_floor = P_in * (f_e + f_sl)
                    
                    if stop_loss_distance < T_floor:
                        continue
                    
                    # Calculate stop loss price (must be below entry for long trades)
                    P_sl = (P_in * (1 + f_e) - stop_loss_distance) / (1 - f_sl)
                    
                    # Calculate risk-reward distance
                    T_rr = stop_loss_distance * T['risk_reward_ratio']
                    
                    # Calculate take profit price (must be above entry for long trades)
                    P_tp = (P_in * (1 + f_e) + T_rr) / (1 - f_tp)

                    P_be_tp = P_in * (1 + f_e) / (1 - f_tp)
                    
                    # Validate that stop loss is below entry and take profit is above stop loss
                    # if P_sl >= P_in or P_tp <= P_sl:
                    #     print(f"WARNING: Invalid price levels - Entry: {P_in:.4f}, SL: {P_sl:.4f}, TP: {P_tp:.4f}")
                    #     print(f"  stop_loss_distance: {stop_loss_distance:.6f}, T_floor: {T_floor:.6f}")
                    #     continue  # Skip this trade

                    size = min(risk_amount / entry_price, available_cash / entry_price)
                    
                    if size <= 0: continue
                    
                    # Calculate trade value with Taker fee (1.2% added to trade value)
                    base_trade_value = entry_price * size
                    taker_fee_amount = base_trade_value * TAKER_FEE
                    trade_value = base_trade_value + taker_fee_amount

                    # print(f"Entry price: {P_in}, Stop loss: {P_sl}, Take profit: {P_tp}, Size: {size}, Trade value: {trade_value}, Bracket size: {T_rr}")
                    
                    # Final check: ensure we don't exceed available cash
                    if trade_value > available_cash:
                        # Reduce size to fit available cash
                        max_base_value = available_cash / (1 + TAKER_FEE)  # Fixed: use (1 + TAKER_FEE) not (1 - TAKER_FEE)
                        size = np.floor(max_base_value / entry_price)
                        if size <= 0:
                            continue
                        base_trade_value = entry_price * size
                        taker_fee_amount = base_trade_value * TAKER_FEE
                        trade_value = base_trade_value + taker_fee_amount
                    
                    # Update cash ledger when opening trade
                    try:
                        cash_ledger.open_trade(trade_value, taker_fee_amount)
                    except ValueError as e:
                        # Skip this trade if insufficient cash
                        continue
                        
                    open_trades.append({
                        'entry_idx': idx,
                        'entry_price': entry_price,
                        'stop_loss': P_sl,
                        'take_profit': P_tp,
                        'size': size,
                        'capital_at_entry': cash_ledger.get_total_capital(),  # Store current capital
                        'trade_value': trade_value,  # Store the value of the trade
                        'predicted_change': row['Predicted_Change'],
                        'risk_percentage': risk_percentage,
                        'result': None,
                        'exit_idx': None,
                        'exit_price': None,
                        'profit': None
                    })
            
            # Calculate total portfolio value (cash + open trade values) for every bar
            open_trades_market_value = 0.0
            for trade in open_trades:
                # Calculate current value of open trade based on current price
                current_price = row['Close']
                trade_current_value = current_price * trade['size']
                # P&L from entry (trade_value already includes Taker fee)
                trade_pnl = trade_current_value - trade['trade_value']
                open_trades_market_value += trade_pnl
            
            total_portfolio_value = cash_ledger.get_portfolio_value(open_trades_market_value)
            
            # Update equity curve (adjust index for trading period only)
            equity_curve.iloc[idx - trading_start_idx] = np.float64(total_portfolio_value)
    
    # Close any remaining open trades
    for trade in open_trades:
        exit_price = df_window.iloc[-1]['Close'] * (1 - SLIPPAGE)
        # Calculate profit with Maker fee (0.6% deducted from sale value)
        gross_profit = (exit_price - trade['entry_price']) * trade['size']
        sale_value = exit_price * trade['size']
        maker_fee_amount = sale_value * MAKER_FEE
        profit = gross_profit - maker_fee_amount
        
        # Update cash ledger
        cash_ledger.close_trade(trade['trade_value'], profit, maker_fee_amount)
        
        trade['exit_idx'] = len(df_window) - 1
        trade['exit_price'] = exit_price
        trade['result'] = 'EOD'
        trade['profit'] = profit
        trade_history.append(trade)
    
    # Update final equity curve value
    if len(equity_curve) > 0:
        # Calculate final portfolio value (cash + any remaining open trades)
        final_portfolio_value = cash_ledger.get_total_capital()
        
        equity_curve.iloc[-1] = np.float64(final_portfolio_value)
    
    # Calculate performance metrics
    results_df = pd.DataFrame(trade_history)
    returns = equity_curve.pct_change().dropna()
    
    # Calculate final portfolio value for metrics
    final_portfolio_value = cash_ledger.get_total_capital()
    
    # Calculate CAGR using the trading period dates
    trading_start_date = df_window['Date'].iloc[trading_start_idx]
    trading_end_date = df_window['Date'].iloc[-1]
    cagr = calculate_cagr(INITIAL_CAPITAL, final_portfolio_value, trading_start_date, trading_end_date)
    
    # Calculate MAR (Managed Account Ratio) = CAGR / |Max Drawdown|
    max_drawdown_decimal = calculate_max_drawdown(equity_curve)
    mar = calculate_mar(cagr, max_drawdown_decimal)
    
    # Handle case where no trades were made
    if len(results_df) == 0:
        metrics = {
            'total_return': 0.0,
            'final_capital': final_portfolio_value,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'cagr': cagr * 100,  # Convert to percentage
            'mar': mar,  # MAR is already a ratio, no conversion needed
            'trade_count': 0,
            'equity_curve': equity_curve,
            'trade_history': results_df
        }
    else:
        metrics = {
            'total_return': (final_portfolio_value / INITIAL_CAPITAL - 1) * 100,
            'final_capital': final_portfolio_value,
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'win_rate': np.mean(results_df['profit'] > 0) * 100,
            'max_drawdown': calculate_max_drawdown(equity_curve) * 100,
            'cagr': cagr * 100,  # Convert to percentage
            'mar': mar,  # MAR is already a ratio, no conversion needed
            'trade_count': len(results_df),
            'equity_curve': equity_curve,
            'trade_history': results_df
        }
    
    return metrics

def save_top_parameters(study, symbol):
    """Save top 10 best parameters to JSON file"""
    # Get all trials
    trials = study.trials
    
    # Filter out failed trials and create a list of successful trials with their scores
    successful_trials = [(t.number, t.value, t.params, t.user_attrs) for t in trials if t.value is not None]
    
    # Sort by score (descending)
    successful_trials.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 10
    top_10 = successful_trials[:10]
    
    # Create parameters dictionary
    params_to_save = {
        'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'top_parameters': []
    }
    
    # Create Visualization directory if it doesn't exist
    os.makedirs('Visualization', exist_ok=True)
    
    # Add each trial's parameters and metrics
    for i, (trial_num, score, params, attrs) in enumerate(top_10):
        # Calculate actual window_size and retrain_interval
        window_size = clamp(int(len(df) * params['window_fraction']), MIN_WINDOW, MAX_WINDOW)
        retrain_interval = max(int(window_size * params['retrain_fraction']), 10)
        
        # Separate trading and model parameters
        trading_params = {
            'min_risk_percentage': params['min_risk_percentage'],
            'max_risk_percentage': params['max_risk_percentage'],
            'risk_scaling_factor': params['risk_scaling_factor'],
            'risk_reward_ratio': params['risk_reward_ratio'],
            'min_predicted_move': params['min_predicted_move'],
            'partial_take_profit': params['partial_take_profit'],
            'min_holding_period': params['min_holding_period'],
            'max_holding_period': params['max_holding_period'],
            'max_concurrent_trades': params['max_concurrent_trades'],
            'stop_loss_atr_multiplier': params['stop_loss_atr_multiplier'],
            'atr_predicted_weight': params['atr_predicted_weight'],
            'aggressiveness': params['aggressiveness'],
            'feature_count_k': params['feature_count_k'],
            'window_fraction': params['window_fraction'],
            'retrain_fraction': params['retrain_fraction'],
            'calculated_window_size': window_size,
            'calculated_retrain_interval': retrain_interval
        }
        
        model_params = {
            'learning_rate': params['lr'],
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'max_leaves': params['max_leaves'],
            'min_child_weight': params['min_child_weight'],
            'gamma': params['gamma'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'colsample_bylevel': params['colsample_bylevel'],
            'reg_lambda': params['reg_lambda'],
            'reg_alpha': params['reg_alpha'],
            'max_bin': params['max_bin']
        }
        
        # Get the equity curve from trial attributes and convert back to Series
        equity_curve_dict = attrs.get('equity_curve')
        if equity_curve_dict is not None:
            equity_curve = pd.Series(
                data=equity_curve_dict['values'],
                index=pd.to_datetime(equity_curve_dict['dates'])
            )
            
            # Plot and save equity curve phases for this trial
            # Adjust figure size based on number of phases
            cols = math.ceil(math.sqrt(EQUITY_CURVE_PHASES))
            rows = math.ceil(EQUITY_CURVE_PHASES / cols)
            plt.figure(figsize=(cols * 4, rows * 3))
            
            # Split the timeline into EQUITY_CURVE_PHASES phases
            total_points = len(equity_curve)
            phase_size = total_points // EQUITY_CURVE_PHASES
            
            # Create subplots for each phase
            for j in range(EQUITY_CURVE_PHASES):
                start_idx = j * phase_size
                end_idx = (j + 1) * phase_size if j < EQUITY_CURVE_PHASES - 1 else total_points
                
                plt.subplot(rows, cols, j + 1)
                phase_data = equity_curve.iloc[start_idx:end_idx]
                plt.plot(phase_data.index, phase_data.values)
                plt.title(f'Phase {j + 1} Equity Curve')
                plt.xlabel('Date')
                plt.ylabel('Capital ($)')
                plt.grid(True)
                # Make x-axis dates smaller to prevent overlapping
                plt.xticks(rotation=45, fontsize=8)
                plt.tick_params(axis='x', labelsize=8)
                
                # Add phase statistics
                phase_returns = phase_data.pct_change().dropna()
                phase_sharpe = calculate_sharpe_ratio(phase_returns)
                phase_drawdown = calculate_max_drawdown(phase_data) * 100
                plt.text(0.02, 0.98, f'Sharpe: {phase_sharpe:.2f}\nMax DD: {phase_drawdown:.2f}%', 
                        transform=plt.gca().transAxes, verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(f'Visualization/trial_{i+1}_equity_curve_phases.png')
            plt.close()
        
        params_to_save['top_parameters'].append({
            'trial_number': trial_num,
            'score': float(score),
            'trading_parameters': trading_params,
            'model_parameters': model_params,
            'metrics': {
                'total_return': attrs.get('total_return', 0.0),
                'sharpe_ratio': attrs.get('sharpe_ratio', 0.0),
                'win_rate': attrs.get('win_rate', 0.0),
                'max_drawdown': attrs.get('max_drawdown', 0.0),
                'cagr': attrs.get('cagr', 0.0),
                'mar': attrs.get('mar', 0.0),
                'trade_count': attrs.get('trade_count', 0)
            }
        })
    
    # Save to JSON
    os.makedirs('Parameters', exist_ok=True)
    params_file = f'Parameters/{symbol}_Joint_Optimization.json'
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=4)
    print(f"\nTop 10 parameters saved to {params_file}")
    print(f"Top 10 equity curve phase graphs saved to Visualization folder")

def objective(trial):
    """Objective function for Optuna joint optimization"""
    # Define trading parameters step by step
    
    T = {
        'min_risk_percentage': trial.suggest_float('min_risk_percentage', 0.10, 0.25),
        'max_risk_percentage': trial.suggest_float('max_risk_percentage', 0.50, 0.75),
        'risk_scaling_factor': trial.suggest_float('risk_scaling_factor', 1.5, 3.0),
        'risk_reward_ratio': trial.suggest_float('risk_reward_ratio', 1.5, 3.0),
        'min_predicted_move': trial.suggest_float('min_predicted_move', 0.005, 0.01),
        'window_fraction': trial.suggest_float('window_fraction', 0.01, 0.5),  # 1% to 50% of the data
        'retrain_fraction': trial.suggest_float('retrain_fraction', 0.05, 1),  # 5% to 100% of the window size
        'partial_take_profit': trial.suggest_float('partial_take_profit', 0.7, 0.95),
        'min_holding_period': trial.suggest_int('min_holding_period', 5, 50),
        'max_holding_period': trial.suggest_int('max_holding_period', 60, 100),
        'max_concurrent_trades': trial.suggest_int('max_concurrent_trades', 1, 10),
        'stop_loss_atr_multiplier': trial.suggest_float('stop_loss_atr_multiplier', 0.5, 4.0),
        'atr_predicted_weight': trial.suggest_float('atr_predicted_weight', 0.0, 1.0),  # 0 = all predicted, 1 = all ATR
        'aggressiveness': trial.suggest_float('aggressiveness', 0.5, 5.0),  # Controls how fast risk scales to max
        'feature_count_k': trial.suggest_int('feature_count_k', 16, 64)  # Number of top features to select
    }
    
    # Calculate derived parameters
    window_size = clamp(int(len(df) * T['window_fraction']), MIN_WINDOW, MAX_WINDOW)
    retrain_interval = max(int(window_size * T['retrain_fraction']), 10)
    T['window_size'] = window_size
    T['retrain_interval'] = retrain_interval

    # Define model hyperparameters (H)
    H = {
        'learning_rate': trial.suggest_float("lr", 0.02, 0.12, log=True),
        'n_estimators': trial.suggest_int("n_estimators", 300, 1400),
        'max_depth': trial.suggest_int("max_depth", 3, 6),
        'max_leaves': trial.suggest_int("max_leaves", 32, 128),
        'min_child_weight': trial.suggest_float("min_child_weight", 0.5, 6.0, log=True),
        'gamma': trial.suggest_float("gamma", 0.0, 3.0),
        'subsample': trial.suggest_float("subsample", 0.6, 0.95),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.4, 0.9),
        'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        'reg_lambda': trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        'reg_alpha': trial.suggest_float("reg_alpha", 0.0, 2.0),
        'max_bin': trial.suggest_int("max_bin", 256, 1024),
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cpu'
    }
    
    # Debug: Print parameters for first few trials to verify they match defaults when hardcoded
    if trial.number < 3:
        print(f"\nTrial {trial.number} parameters:")
        print(f"  Trading Parameters (T):")
        print(f"    min_risk: {T['min_risk_percentage']:.6f}")
        print(f"    max_risk: {T['max_risk_percentage']:.6f}")
        print(f"    scaling_factor: {T['risk_scaling_factor']:.6f}")
        print(f"    reward_ratio: {T['risk_reward_ratio']:.6f}")
        print(f"    min_predicted_move: {T['min_predicted_move']:.6f}")
        print(f"    window_size: {T['window_size']}")
        print(f"    retrain_interval: {T['retrain_interval']}")
        print(f"    stop_loss_atr_multiplier: {T['stop_loss_atr_multiplier']:.6f}")
        print(f"    atr_predicted_weight: {T['atr_predicted_weight']:.6f}")
        print(f"    aggressiveness: {T['aggressiveness']:.6f}")
        print(f"  Model Parameters (H):")
        print(f"    learning_rate: {H['learning_rate']:.6f}")
        print(f"    n_estimators: {H['n_estimators']}")
        print(f"    max_depth: {H['max_depth']}")
        print(f"    max_leaves: {H['max_leaves']}")
    
    # Filter data dynamically based on the actual window_size for this trial
    # This ensures we use exactly the right amount of data for each trial
    start_date = '2025-01-15'
    end_date = '2025-09-09'
    trial_buffered_start = pd.to_datetime(start_date) - pd.Timedelta(days=T['window_size']*15/60/24)
    df_trial = df[(df['Date'] >= trial_buffered_start) & (df['Date'] <= end_date)]
    
    # if trial.number < 3:
    #     print(f"  Trial data length: {len(df_trial)} bars")
    #     print(f"  Trial date range: {df_trial['Date'].min()} to {df_trial['Date'].max()}")
    #     print(f"  Required window size: {T['window_size']}")
    #     print(f"  Sufficient data: {'YES' if len(df_trial) >= T['window_size'] else 'NO'}")
    
    # Run strategy with current parameters and trial-specific data
    metrics = run_strategy(df_trial, T, H, feature_cols, target_cols)
    
    # If no trades were made, prune the trial
    if metrics['trade_count'] == 0:
        raise optuna.TrialPruned()
    
    # Store metrics as trial attributes
    trial.set_user_attr('total_return', metrics['total_return'])
    trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
    trial.set_user_attr('win_rate', metrics['win_rate'])
    trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
    trial.set_user_attr('cagr', metrics['cagr'])
    trial.set_user_attr('mar', metrics['mar'])
    trial.set_user_attr('trade_count', metrics['trade_count'])
    
    # Convert equity curve to JSON serializable format
    equity_curve_dict = {
        'dates': metrics['equity_curve'].index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'values': metrics['equity_curve'].values.tolist()
    }
    trial.set_user_attr('equity_curve', equity_curve_dict)
    
    # Calculate composite score that balances multiple objectives
    composite_score = calculate_composite_score(metrics['mar'], metrics['win_rate'], T['risk_reward_ratio'], metrics['trade_count'], start_date, end_date)
    
    # Save top parameters after each trial
    save_top_parameters(study, symbol)
    
    return composite_score


if __name__ == "__main__":
    # Load and prepare data
    data_path = f"DB/{symbol}_fifteenminute_indicators.csv"
    df, feature_cols, target_cols = preprocess_data(pd.read_csv(data_path))
    
    #Filter data by time range
    start_date = '2025-01-15'  # Format: 'YYYY-MM-DD'
    end_date = '2025-09-09'    # Format: 'YYYY-MM-DD'

    # For default mode, use DEFAULT_WINDOW_SIZE for buffering
    if SKIP_OPTIMIZATION:
        buffered_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=T_default['window_size']*15/60/24)
        df = df[(df['Date'] >= buffered_start_date) & (df['Date'] <= end_date)]
        print(f"Default mode: Data filtered with DEFAULT_WINDOW_SIZE={T_default['window_size']}")
        print(f"Buffered start date: {buffered_start_date}")
        print(f"Final data length: {len(df)} bars")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    else:
        # For optimization mode, we'll need to filter data dynamically in the objective function
        # For now, use MAX_WINDOW to ensure we have enough data for any possible window size
        buffered_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=MAX_WINDOW*15/60/24)
        df = df[(df['Date'] >= buffered_start_date) & (df['Date'] <= end_date)]
        print(f"Optimization mode: Data filtered with MAX_WINDOW={MAX_WINDOW}")
        print(f"Buffered start date: {buffered_start_date}")
        print(f"Final data length: {len(df)} bars")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print("Note: Data will be further filtered in objective function based on actual window_size")
    
    # Run optimization or use default parameters
    if SKIP_OPTIMIZATION:
        print("\nUsing default parameters (optimization skipped)")
        print(f"Min Risk: {T_default['min_risk_percentage']:.3f}")
        print(f"Max Risk: {T_default['max_risk_percentage']:.3f}")
        print(f"Risk Scaling: {T_default['risk_scaling_factor']}")
        print(f"Risk:Reward: {T_default['risk_reward_ratio']}")
        print(f"Min Predicted Move: {T_default['min_predicted_move']:.3f}")
        print(f"Partial Take Profit: {T_default['partial_take_profit']:.3f}")
        print(f"Min Holding Period: {T_default['min_holding_period']}")
        print(f"Max Holding Period: {T_default['max_holding_period']}")
        print(f"Max Concurrent Trades: {T_default['max_concurrent_trades']}")
        print(f"Window Size: {T_default['window_size']}")
        print(f"Retrain Interval: {T_default['retrain_interval']}")
        print(f"Stop Loss ATR Multiplier: {T_default['stop_loss_atr_multiplier']}")
        print(f"ATR vs Predicted Weight: {T_default['atr_predicted_weight']}")
        print(f"Maker Fee (Buy): {MAKER_FEE*100:.1f}%")
        print(f"Taker Fee (Sell): {TAKER_FEE*100:.1f}%")
        print(f"Total Round-trip Cost: {(MAKER_FEE + TAKER_FEE)*100:.1f}%")
        print("Stop loss and take profit levels are adjusted to compensate for fees")

        
        best_metrics = run_strategy(df, T_default, H_default, feature_cols, target_cols)
        
        # Calculate composite score for default parameters
        composite_score = calculate_composite_score(best_metrics['mar'], best_metrics['win_rate'], T_default['risk_reward_ratio'], best_metrics['trade_count'], start_date, end_date)
        
        results = [{
            # Trading parameters
            'min_risk_percentage': T_default['min_risk_percentage'],
            'max_risk_percentage': T_default['max_risk_percentage'],
            'risk_scaling_factor': T_default['risk_scaling_factor'],
            'risk_reward_ratio': T_default['risk_reward_ratio'],
            'min_predicted_move': T_default['min_predicted_move'],
            'stop_loss_atr_multiplier': T_default['stop_loss_atr_multiplier'],
            'atr_predicted_weight': T_default['atr_predicted_weight'],
            'window_size': T_default['window_size'],
            'retrain_interval': T_default['retrain_interval'],
            'partial_take_profit': T_default['partial_take_profit'],
            'min_holding_period': T_default['min_holding_period'],
            'max_holding_period': T_default['max_holding_period'],
            'max_concurrent_trades': T_default['max_concurrent_trades'],
            'aggressiveness': T_default['aggressiveness'],
            'feature_count_k': T_default['feature_count_k'],
            # Model parameters
            'learning_rate': H_default['learning_rate'],
            'n_estimators': H_default['n_estimators'],
            'max_depth': H_default['max_depth'],
            'max_leaves': H_default['max_leaves'],
            'min_child_weight': H_default['min_child_weight'],
            'gamma': H_default['gamma'],
            'subsample': H_default['subsample'],
            'colsample_bytree': H_default['colsample_bytree'],
            'colsample_bylevel': H_default['colsample_bylevel'],
            'reg_lambda': H_default['reg_lambda'],
            'reg_alpha': H_default['reg_alpha'],
            'max_bin': H_default['max_bin'],
            # Performance metrics
            'total_return': best_metrics['total_return'],
            'final_capital': best_metrics['final_capital'],
            'sharpe_ratio': best_metrics['sharpe_ratio'],
            'win_rate': best_metrics['win_rate'],
            'max_drawdown': best_metrics['max_drawdown'],
            'cagr': best_metrics['cagr'],
            'mar': best_metrics['mar'],
            'trade_count': best_metrics['trade_count'],
            'composite_score': composite_score,
            'equity_curve': best_metrics['equity_curve'],
            'trade_history': best_metrics['trade_history']
        }]
    else:
        print("\nRunning Optuna optimization...")
        
        # Create storage for the study
        storage_name = f"sqlite:///DB/{symbol}_joint_study.db"
        
        # Handle study resumption
        if RESUME_STUDY is None:
            # Check if study exists
            try:
                study = optuna.load_study(study_name=symbol, storage=storage_name)
                print(f"\nFound existing study with {len(study.trials)} trials. Resuming...")
            except:
                print("\nNo existing study found. Starting new study...")
                study = optuna.create_study(
                    study_name=symbol,
                    storage=storage_name,
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=5)
                )
        elif RESUME_STUDY:
            try:
                study = optuna.load_study(study_name=symbol, storage=storage_name)
                print(f"\nResuming existing study with {len(study.trials)} trials...")
            except:
                print("\nNo existing study found. Starting new study...")
                study = optuna.create_study(
                    study_name=symbol,
                    storage=storage_name,
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=5)
                )
        else:
            print("\nStarting new study...")
            study = optuna.create_study(
                study_name=symbol,
                storage=storage_name,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=5)
            )
        
        # Calculate remaining trials
        completed_trials = len(study.trials)
        remaining_trials = max(0, OPTIMIZATION_TRIALS - completed_trials)
        
        if remaining_trials > 0:
            print(f"\nRunning {remaining_trials} remaining trials...")
            study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True)
        else:
            print("\nAll trials completed!")
        
        # Get best parameters
        best_params = study.best_params
        # Calculate actual window_size and retrain_interval from fractions
        window_size = clamp(int(len(df) * best_params['window_fraction']), MIN_WINDOW, MAX_WINDOW)
        retrain_interval = max(int(window_size * best_params['retrain_fraction']), 10)
        
        # Create T and H dictionaries from best parameters
        T_best = {
            'min_risk_percentage': best_params['min_risk_percentage'],
            'max_risk_percentage': best_params['max_risk_percentage'],
            'risk_scaling_factor': best_params['risk_scaling_factor'],
            'risk_reward_ratio': best_params['risk_reward_ratio'],
            'min_predicted_move': best_params['min_predicted_move'],
            'partial_take_profit': best_params['partial_take_profit'],
            'min_holding_period': best_params['min_holding_period'],
            'max_holding_period': best_params['max_holding_period'],
            'max_concurrent_trades': best_params['max_concurrent_trades'],
            'stop_loss_atr_multiplier': best_params['stop_loss_atr_multiplier'],
            'atr_predicted_weight': best_params['atr_predicted_weight'],
            'window_size': window_size,
            'retrain_interval': retrain_interval
        }
        
        H_best = {
            'learning_rate': best_params['lr'],
            'n_estimators': best_params['n_estimators'],
            'max_depth': best_params['max_depth'],
            'max_leaves': best_params['max_leaves'],
            'min_child_weight': best_params['min_child_weight'],
            'gamma': best_params['gamma'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'colsample_bylevel': best_params['colsample_bylevel'],
            'reg_lambda': best_params['reg_lambda'],
            'reg_alpha': best_params['reg_alpha'],
            'max_bin': best_params['max_bin'],
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'cpu'
        }
        
        # Filter data for the best parameters using the same logic as in objective function
        best_buffered_start = pd.to_datetime(start_date) - pd.Timedelta(days=window_size*15/60/24)
        df_best = df[(df['Date'] >= best_buffered_start) & (df['Date'] <= end_date)]
        
        print(f"\nBest parameters data filtering:")
        print(f"  Window size: {window_size}")
        print(f"  Buffered start: {best_buffered_start}")
        print(f"  Data length: {len(df_best)} bars")
        print(f"  Date range: {df_best['Date'].min()} to {df_best['Date'].max()}")
        
        best_metrics = run_strategy(df_best, T_best, H_best, feature_cols, target_cols)
        
        
        results = [{
            # Trading parameters
            'min_risk_percentage': best_params['min_risk_percentage'],
            'max_risk_percentage': best_params['max_risk_percentage'],
            'risk_scaling_factor': best_params['risk_scaling_factor'],
            'risk_reward_ratio': best_params['risk_reward_ratio'],
            'min_predicted_move': best_params['min_predicted_move'],
            'window_fraction': best_params['window_fraction'],
            'retrain_fraction': best_params['retrain_fraction'],
            'stop_loss_atr_multiplier': best_params['stop_loss_atr_multiplier'],
            'atr_predicted_weight': best_params['atr_predicted_weight'],
            'aggressiveness': best_params['aggressiveness'],
            'feature_count_k': best_params['feature_count_k'],
            'window_size': window_size,
            'retrain_interval': retrain_interval,
            'partial_take_profit': best_params['partial_take_profit'],
            'min_holding_period': best_params['min_holding_period'],
            'max_holding_period': best_params['max_holding_period'],
            'max_concurrent_trades': best_params['max_concurrent_trades'],
            # Model parameters
            'learning_rate': best_params['lr'],
            'n_estimators': best_params['n_estimators'],
            'max_depth': best_params['max_depth'],
            'max_leaves': best_params['max_leaves'],
            'min_child_weight': best_params['min_child_weight'],
            'gamma': best_params['gamma'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'colsample_bylevel': best_params['colsample_bylevel'],
            'reg_lambda': best_params['reg_lambda'],
            'reg_alpha': best_params['reg_alpha'],
            'max_bin': best_params['max_bin'],
            # Performance metrics
            'total_return': best_metrics['total_return'],
            'final_capital': best_metrics['final_capital'],
            'sharpe_ratio': best_metrics['sharpe_ratio'],
            'win_rate': best_metrics['win_rate'],
            'max_drawdown': best_metrics['max_drawdown'],
            'cagr': best_metrics['cagr'],
            'mar': best_metrics['mar'],
            'trade_count': best_metrics['trade_count'],
            'equity_curve': best_metrics['equity_curve'],
            'trade_history': best_metrics['trade_history']
        }]

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Get best parameters
    best_params = results_df.iloc[0]
    print("\nBest Joint Optimization Parameters:")
    print("\nTrading Parameters:")
    print(f"Min Risk: {best_params['min_risk_percentage']:.3f}")
    print(f"Max Risk: {best_params['max_risk_percentage']:.3f}")
    print(f"Risk Scaling: {best_params['risk_scaling_factor']:.2f}")
    print(f"Risk:Reward: {best_params['risk_reward_ratio']:.2f}")
    print(f"Min Predicted Move: {best_params['min_predicted_move']:.3f}")
    print(f"Stop Loss ATR Multiplier: {best_params['stop_loss_atr_multiplier']:.2f}")
    print(f"ATR vs Predicted Weight: {best_params['atr_predicted_weight']:.2f}")
    print(f"Aggressiveness: {best_params['aggressiveness']:.2f}")
    print(f"Feature Count K: {best_params['feature_count_k']}")
    print(f"Window Size: {best_params['window_size']}")
    print(f"Retrain Interval: {best_params['retrain_interval']}")
    print(f"Partial Take Profit: {best_params['partial_take_profit']:.3f}")
    print(f"Min Holding Period: {best_params['min_holding_period']}")
    print(f"Max Holding Period: {best_params['max_holding_period']}")
    print(f"Max Concurrent Trades: {best_params['max_concurrent_trades']}")
    print("\nModel Parameters:")
    print(f"Learning Rate: {best_params['learning_rate']:.4f}")
    print(f"N Estimators: {best_params['n_estimators']}")
    print(f"Max Depth: {best_params['max_depth']}")
    print(f"Max Leaves: {best_params['max_leaves']}")
    print(f"Min Child Weight: {best_params['min_child_weight']:.3f}")
    print(f"Gamma: {best_params['gamma']:.3f}")
    print(f"Subsample: {best_params['subsample']:.3f}")
    print(f"Colsample by Tree: {best_params['colsample_bytree']:.3f}")
    print(f"Colsample by Level: {best_params['colsample_bylevel']:.3f}")
    print(f"Reg Lambda: {best_params['reg_lambda']:.3f}")
    print(f"Reg Alpha: {best_params['reg_alpha']:.3f}")
    print(f"Max Bin: {best_params['max_bin']}")
    print(f"\nPerformance Metrics:")
    print(f"Total Return: {best_params['total_return']:.2f}%")
    print(f"Final Capital: ${best_params['final_capital']:,.2f}")
    print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    print(f"Win Rate: {best_params['win_rate']:.2f}%")
    print(f"Max Drawdown: {best_params['max_drawdown']:.2f}%")
    print(f"CAGR: {best_params['cagr']:.2f}%")
    print(f"MAR: {best_params['mar']:.2f}")
    print(f"Trade Count: {best_params['trade_count']}")
    print(f"Composite Score: {best_params['composite_score']:.2f}")

    # Plot best equity curve
    # Adjust figure size based on number of phases
    cols = math.ceil(math.sqrt(EQUITY_CURVE_PHASES))
    rows = math.ceil(EQUITY_CURVE_PHASES / cols)
    plt.figure(figsize=(cols * 4, rows * 3))

    # Split the timeline into EQUITY_CURVE_PHASES phases
    total_points = len(best_params['equity_curve'])
    phase_size = total_points // EQUITY_CURVE_PHASES

    # Create subplots for each phase
    for i in range(EQUITY_CURVE_PHASES):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < EQUITY_CURVE_PHASES - 1 else total_points
        
        plt.subplot(rows, cols, i + 1)
        phase_data = best_params['equity_curve'].iloc[start_idx:end_idx]
        plt.plot(phase_data.index, phase_data.values)
        plt.title(f'Phase {i + 1} Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.grid(True)
        # Make x-axis dates smaller to prevent overlapping
        plt.xticks(rotation=45, fontsize=8)
        plt.tick_params(axis='x', labelsize=8)
        
        # Add phase statistics
        phase_returns = phase_data.pct_change().dropna()
        phase_sharpe = calculate_sharpe_ratio(phase_returns)
        phase_drawdown = calculate_max_drawdown(phase_data) * 100
        plt.text(0.02, 0.98, f'Sharpe: {phase_sharpe:.2f}\nMax DD: {phase_drawdown:.2f}%', 
                transform=plt.gca().transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('DB/charts/best_equity_curve_phases.png')
    print("\nPhase-wise equity curves saved to DB/charts/best_equity_curve_phases.png")

    # Also save the original full equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(best_params['equity_curve'].index, best_params['equity_curve'].values)
    plt.title('Best Parameter Set Equity Curve (Full Timeline)')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    # Make x-axis dates smaller to prevent overlapping
    plt.xticks(rotation=45, fontsize=8)
    plt.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    plt.savefig('DB/charts/best_equity_curve_full.png')
    print("Full equity curve saved to DB/charts/best_equity_curve_full.png")

    # Save results
    results_df.to_csv('DB/joint_optimization_results.csv', index=False)
    best_params['trade_history'].to_csv('DB/best_joint_parameter_trade_history.csv', index=False)
    print("\nResults saved to DB/joint_optimization_results.csv")
    print("Best trade history saved to DB/best_joint_parameter_trade_history.csv")


