import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
import optuna
import json
import math
from utils import preprocess_data, calculate_feature_importance, clamp, calculate_sharpe_ratio, calculate_max_drawdown, calculate_distribution_metrics, calculate_cagr, calculate_mar
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from Optimization.FAPT_Wasserstein import predict_optimal_parameters, get_top_market_weather_features

import scipy.stats


symbol = "XRP_USD"

# Default parameters (used when skip_optimization=True)
DEFAULT_MIN_RISK = 0.018193997619464653
DEFAULT_MAX_RISK = 0.8396903321522786
DEFAULT_SCALING = 2.3336616807215136
DEFAULT_RR = 1.5016122471326743
DEFAULT_MIN_PREDICTED_MOVE = 0.005653821215830177
DEFAULT_MIN_HOLDING_PERIOD = 9
DEFAULT_MAX_HOLDING_PERIOD = 10
DEFAULT_PARTIAL_TAKE_PROFIT = 0.7879261282151514
DEFAULT_MAX_CONCURRENT_TRADES = 9
DEFAULT_WINDOW_SIZE = 25000
DEFAULT_RETREIN_INTERVAL = 25000
DEFAULT_STOP_LOSS_ATR_MULTIPLIER = 3.8680612209950582
DEFAULT_ATR_PREDICTED_WEIGHT = 0.8151555981889735  # Weight for ATR vs predicted move (0.6 = 60% ATR, 40% predicted)

# Fixed parameters
INITIAL_CAPITAL = 2000
SLIPPAGE = 0.000
MAKER_FEE = 0.006  # 0.6% fee when buying (adding to trade value)
TAKER_FEE = 0.012  # 1.2% fee when selling (deducted from sale value)


# Flag to skip optimization
SKIP_OPTIMIZATION = False  # Set to True to use default parameters
USE_FAPT = False
OPTIMIZATION_TRIALS = 6000
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


def run_strategy(df_window, min_risk_percentage, max_risk_percentage, risk_scaling_factor, risk_reward_ratio, min_predicted_move, window_size, retrain_interval, partial_take_profit, min_holding_period, max_holding_period, max_concurrent_trades, feature_cols, target_cols, stop_loss_atr_multiplier=1.5, atr_predicted_weight=0.6, model_params=None):
    """Run trading strategy with given parameters"""
    
    # Initialize predicted change column
    df_window['Predicted_Change'] = np.nan
    
    # Simulate trading with model retraining every retrain_interval bars
    capital = INITIAL_CAPITAL
    open_trades = []
    trade_history = []
    holdout_predictions = []
    holdout_actuals = []
    rolling_metrics = []
    # Initialize equity curve only for the trading period (after initial training window)
    trading_start_idx = window_size
    equity_curve = pd.Series(index=df_window['Date'].iloc[trading_start_idx:], data=np.nan, dtype=np.float64)
    
    n = len(df_window)
    
    # Initialize model and feature importance for first window
    print("\nInitializing first window...")
    initial_window = df_window.iloc[:window_size].copy()
    current_features = calculate_feature_importance(
        initial_window, 
        feature_cols, 
        target_cols,
        iterations=1,
        save_importance=False,
        visualize_importance=False
    )

    # Use provided model_params or default parameters
    # if model_params is None:
    #     model_params = {
    #         'n_estimators': 50,
    #         'max_depth': 4,
    #         'learning_rate': 0.1,
    #         'min_child_weight': 10,
    #         'subsample': 0.7,
    #         'colsample_bytree': 0.7,
    #         'reg_alpha': 0.5,
    #         'reg_lambda': 1.0,
    #         'random_state': 42,
    #         'tree_method': 'hist',
    #         'device': 'cpu'
    #     }
    
    # Ensure required parameters are set
    model_params.setdefault('random_state', 42)
    model_params.setdefault('tree_method', 'hist')
    model_params.setdefault('device', 'cpu')
    
    model = xgb.XGBRegressor(**model_params)
    
    # Scale initial window data
    X_initial = df_window.iloc[:window_size][current_features].copy()
    for col in current_features:
        # Use expanding mean/std for scaling with shift to prevent look-ahead bias
        mean = X_initial[col].expanding(min_periods=1).mean().shift(1)
        std = X_initial[col].expanding(min_periods=1).std().shift(1)
        X_initial[col] = (X_initial[col] - mean) / (std + 1e-8)
    
    y_initial = df_window.iloc[:window_size][target_cols].values.ravel()
    model.fit(X_initial, y_initial)
    
    # Process data in chunks
    print("\nProcessing data in chunks...")
    total_data_processed = 0
    
    for start in range(window_size, n, retrain_interval):
        end = min(start + retrain_interval, n)
        total_data_processed += retrain_interval
        
        # Recalculate feature importance every window_size worth of data
        if total_data_processed >= retrain_interval:
            print(f"\nRecalculating feature importance at index {start}...")
            feature_selection_data = df_window.iloc[start-window_size:start].copy()
            current_features = calculate_feature_importance(
                feature_selection_data,
                feature_cols,
                target_cols,
                iterations=1,
                save_importance=False,
                visualize_importance=False
            )
            total_data_processed = 0
        
        current_holdout = df_window.iloc[start:end].copy()
        current_train = df_window.iloc[:start].copy()

        # Prepare features for current training data
        current_X_train = current_train[current_features].copy()
        current_y_train = current_train[target_cols].values.ravel()
        
        # Prepare features for current holdout chunk
        current_X_holdout = current_holdout[current_features].copy()
        current_y_holdout = current_holdout[target_cols].values.ravel()
        
        # Scale features using rolling statistics
        for col in current_features:
            mean = current_X_train[col].expanding(min_periods=1).mean().shift(1)
            std = current_X_train[col].expanding(min_periods=1).std().shift(1)
            current_X_train[col] = (current_X_train[col] - mean) / (std + 1e-8)
            last_mean = mean.iloc[-1]
            last_std = std.iloc[-1]
            current_X_holdout[col] = (current_X_holdout[col] - last_mean) / (last_std + 1e-8)
        
        # Train model on current training data
        current_model = xgb.XGBRegressor(**model_params)
        current_model.fit(current_X_train, current_y_train)

        # Make predictions on current holdout chunk
        chunk_predictions = current_model.predict(current_X_holdout)
        df_window.loc[df_window.index[start:end], 'Predicted_Change'] = chunk_predictions

        holdout_predictions.extend(chunk_predictions)
        holdout_actuals.extend(current_y_holdout)    

        # Calculate metrics for this chunk
        chunk_mse = mean_squared_error(current_y_holdout, chunk_predictions)
        chunk_r2 = r2_score(current_y_holdout, chunk_predictions)
        chunk_dir_acc = np.mean((current_y_holdout > 0) == (chunk_predictions > 0))
        
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
                    # Calculate profit with Taker fee (1.2% deducted from sale value)
                    gross_profit = (exit_price - trade['entry_price']) * trade['size']
                    sale_value = exit_price * trade['size']
                    taker_fee_amount = sale_value * TAKER_FEE
                    profit = gross_profit - taker_fee_amount
                    # Add back the trade value plus profit/loss
                    capital += profit
                    trade['exit_idx'] = idx
                    trade['exit_price'] = exit_price
                    trade['result'] = 'TP'
                    trade['profit'] = profit
                    trade_history.append(trade)
                    closed_trades.append(trade)
                elif low <= trade['stop_loss']:
                    exit_price = trade['stop_loss'] * (1 - SLIPPAGE)
                    # Calculate profit with Taker fee (1.2% deducted from sale value)
                    gross_profit = (exit_price - trade['entry_price']) * trade['size']
                    sale_value = exit_price * trade['size']
                    taker_fee_amount = sale_value * TAKER_FEE
                    profit = gross_profit - taker_fee_amount
                    # Add back the trade value plus profit/loss
                    capital += profit
                    trade['exit_idx'] = idx
                    trade['exit_price'] = exit_price
                    trade['result'] = 'SL'
                    trade['profit'] = profit
                    trade_history.append(trade)
                    closed_trades.append(trade)
                elif holding_period >= min_holding_period:
                    projected_tp = trade['take_profit']
                    projected_entry = trade['entry_price']
                    # Calculate partial take profit with fee compensation
                    # The partial TP should be adjusted to account for the taker fee
                    taker_fee_factor = 1 - TAKER_FEE
                    tp_partial = projected_entry + partial_take_profit * (projected_tp - projected_entry) / taker_fee_factor
                    if high >= tp_partial:
                        exit_price = tp_partial * (1 - SLIPPAGE)
                        # Calculate profit with Taker fee (1.2% deducted from sale value)
                        gross_profit = (exit_price - trade['entry_price']) * trade['size']
                        sale_value = exit_price * trade['size']
                        taker_fee_amount = sale_value * TAKER_FEE
                        profit = gross_profit - taker_fee_amount
                        # Add back the trade value plus profit/loss
                        capital += profit
                        trade['exit_idx'] = idx
                        trade['exit_price'] = exit_price
                        trade['result'] = 'Partial TP'
                        trade['profit'] = profit
                        trade_history.append(trade)
                        closed_trades.append(trade)
                    elif low <= projected_entry:
                        exit_price = projected_entry * (1 - SLIPPAGE)
                        # Calculate profit with Taker fee (1.2% deducted from sale value)
                        gross_profit = (exit_price - trade['entry_price']) * trade['size']
                        sale_value = exit_price * trade['size']
                        taker_fee_amount = sale_value * TAKER_FEE
                        profit = gross_profit - taker_fee_amount
                        # Add back the trade value plus profit/loss
                        capital += profit
                        trade['exit_idx'] = idx
                        trade['exit_price'] = exit_price
                        trade['result'] = 'BE'
                        trade['profit'] = profit
                        trade_history.append(trade)
                        closed_trades.append(trade)
                
                if holding_period >= max_holding_period and trade not in closed_trades:
                    exit_price = row['Close'] * (1 - SLIPPAGE)
                    # Calculate profit with Taker fee (1.2% deducted from sale value)
                    gross_profit = (exit_price - trade['entry_price']) * trade['size']
                    sale_value = exit_price * trade['size']
                    taker_fee_amount = sale_value * TAKER_FEE
                    profit = gross_profit - taker_fee_amount
                    # Add back the trade value plus profit/loss
                    capital += profit
                    trade['exit_idx'] = idx
                    trade['exit_price'] = exit_price
                    trade['result'] = 'MAXHOLD'
                    trade['profit'] = profit
                    trade_history.append(trade)
                    closed_trades.append(trade)
            
            open_trades = [t for t in open_trades if t not in closed_trades]
            
            # Entry logic
            if len(open_trades) < max_concurrent_trades:
                if row['Predicted_Change'] < -min_predicted_move:
                    entry_price = row['Open'] * (1 + SLIPPAGE)
                    
                    predicted_move = abs(row['Predicted_Change'])
                    risk_percentage = min(
                        min_risk_percentage * (1 + (predicted_move / min_predicted_move) * risk_scaling_factor),
                        max_risk_percentage
                    )
                    # scaling_factor = min(1.0, capital / BREAK_EVEN_CAPITAL)  # 0 to 1
                    # adjusted_risk_pct = risk_percentage / scaling_factor
                    
                    # Calculate available cash (capital minus the value of open trades)
                    # This simulates realistic cash management where each trade uses remaining cash
                    available_cash = capital
                    total_open_trade_value = 0
                    for trade in open_trades:
                        # Subtract the trade value (including maker fee) from available cash
                        available_cash -= trade['trade_value']
                        total_open_trade_value += trade['trade_value']
                    
                    # Only proceed if we have enough cash for this trade
                    if available_cash <= 0:
                        continue
                    
                    # Use available cash instead of total capital for risk calculation
                    risk_amount = available_cash * risk_percentage
                    # Calculate stop loss and take profit using hybrid ATR and predicted move approach
                    atr_value = row['ATR'] if 'ATR' in row else row['Close'] * 0.01  # Fallback to 1% if ATR not available
                    
                    # Calculate stop loss distance using hybrid approach
                    atr_stop_distance = atr_value * stop_loss_atr_multiplier
                    predicted_stop_distance = entry_price * predicted_move  # Convert predicted move to price distance
                    
                    # Weighted combination of ATR and predicted move
                    stop_loss_distance = (atr_predicted_weight * atr_stop_distance + 
                                        (1 - atr_predicted_weight) * predicted_stop_distance)
                    
                    # Calculate fee compensation factors
                    # For stop loss: we need to account for the fact that we already paid the maker fee
                    # For take profit: we need to account for both maker fee (already paid) and taker fee (will be paid)
                    maker_fee_factor = 1 + MAKER_FEE  # Factor to account for maker fee already paid
                    taker_fee_factor = 1 - TAKER_FEE  # Factor to account for taker fee on exit
                    
                    # Adjust stop loss to compensate for maker fee (we already paid it, so we need less distance)
                    stop_loss = entry_price - (stop_loss_distance / maker_fee_factor)
                    
                    # Adjust take profit to compensate for both maker and taker fees
                    # We need to reach a higher price to achieve the desired net profit
                    target_net_profit = stop_loss_distance * risk_reward_ratio
                    # Calculate the gross price needed to achieve target net profit after fees
                    take_profit = entry_price + (target_net_profit / (maker_fee_factor * taker_fee_factor))
                    risk_per_share = entry_price - stop_loss
                    
                    if risk_per_share <= 0 or np.isnan(risk_per_share):
                        continue
                        
                    size = risk_amount / risk_per_share
                    # Also limit size by available cash (not total capital)
                    size = min(size, available_cash / entry_price)
                    size = np.floor(size)
                    
                    if size <= 0:
                        continue
                    
                    # Calculate trade value with Maker fee (0.6% added to trade value)
                    base_trade_value = entry_price * size
                    maker_fee_amount = base_trade_value * MAKER_FEE
                    trade_value = base_trade_value + maker_fee_amount
                    
                    # Final check: ensure we don't exceed available cash
                    if trade_value > available_cash:
                        # Reduce size to fit available cash
                        max_base_value = available_cash / (1 + MAKER_FEE)
                        size = np.floor(max_base_value / entry_price)
                        if size <= 0:
                            continue
                        base_trade_value = entry_price * size
                        maker_fee_amount = base_trade_value * MAKER_FEE
                        trade_value = base_trade_value + maker_fee_amount                    
                        
                    open_trades.append({
                        'entry_idx': idx,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': size,
                        'capital_at_entry': capital,  # Store original capital before deduction
                        'trade_value': trade_value,  # Store the value of the trade
                        'predicted_change': row['Predicted_Change'],
                        'risk_percentage': risk_percentage,
                        'result': None,
                        'exit_idx': None,
                        'exit_price': None,
                        'profit': None
                    })
            
            # Calculate total portfolio value (cash + open trade values) for every bar
            total_portfolio_value = capital
            for trade in open_trades:
                # Calculate current value of open trade based on current price
                current_price = row['Close']
                trade_current_value = current_price * trade['size']
                # P&L from entry (trade_value already includes Maker fee)
                trade_pnl = trade_current_value - trade['trade_value']
                total_portfolio_value += trade_pnl
            
            # Update equity curve (adjust index for trading period only)
            equity_curve.iloc[idx - trading_start_idx] = np.float64(total_portfolio_value)
    
    # Close any remaining open trades
    for trade in open_trades:
        exit_price = df_window.iloc[-1]['Close'] * (1 - SLIPPAGE)
        # Calculate profit with Taker fee (1.2% deducted from sale value)
        gross_profit = (exit_price - trade['entry_price']) * trade['size']
        sale_value = exit_price * trade['size']
        taker_fee_amount = sale_value * TAKER_FEE
        profit = gross_profit - taker_fee_amount
        # Add back the trade value plus profit/loss
        capital += profit
        trade['exit_idx'] = len(df_window) - 1
        trade['exit_price'] = exit_price
        trade['result'] = 'EOD'
        trade['profit'] = profit
        trade_history.append(trade)
    
    # Update final equity curve value
    if len(equity_curve) > 0:
        # Calculate final portfolio value (cash + any remaining open trades)
        final_portfolio_value = capital
        for trade in open_trades:
            # Calculate final value of any remaining open trades
            final_price = df_window.iloc[-1]['Close']
            trade_final_value = final_price * trade['size']
            # P&L from entry (trade_value already includes Maker fee)
            trade_pnl = trade_final_value - trade['trade_value']
            final_portfolio_value += trade_pnl
        
        equity_curve.iloc[-1] = np.float64(final_portfolio_value)
    
    # Calculate performance metrics
    results_df = pd.DataFrame(trade_history)
    returns = equity_curve.pct_change().dropna()

    # Calculate final portfolio value for metrics
    final_portfolio_value = capital
    for trade in open_trades:
        final_price = df_window.iloc[-1]['Close']
        trade_final_value = final_price * trade['size']
        # P&L from entry (trade_value already includes Maker fee)
        trade_pnl = trade_final_value - trade['trade_value']
        final_portfolio_value += trade_pnl
    
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
            'model_parameters': {
                'learning_rate': params['learning_rate'],
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
            },
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
    params_file = f'Parameters/{symbol}_Model_Optimization.json'
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=4)
    print(f"\nTop 10 parameters saved to {params_file}")
    print(f"Top 10 equity curve phase graphs saved to Visualization folder")

def objective(trial):
    """Objective function for Optuna optimization - optimizing XGBoost model parameters"""
    # Define XGBoost model parameter ranges
    model_params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.12),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'max_leaves': trial.suggest_int('max_leaves', 32, 128),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 8.0),
        'gamma': trial.suggest_float('gamma', 0.0, 4.0),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'max_bin': trial.suggest_int('max_bin', 256, 1024),
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cpu'
    }
    
    # Run strategy with default trading parameters but optimized model parameters
    metrics = run_strategy(df, 
                           DEFAULT_MIN_RISK, 
                           DEFAULT_MAX_RISK, 
                           DEFAULT_SCALING, 
                           DEFAULT_RR, 
                           DEFAULT_MIN_PREDICTED_MOVE, 
                           DEFAULT_WINDOW_SIZE, 
                           DEFAULT_RETREIN_INTERVAL, 
                           DEFAULT_PARTIAL_TAKE_PROFIT, 
                           DEFAULT_MIN_HOLDING_PERIOD, 
                           DEFAULT_MAX_HOLDING_PERIOD, 
                           DEFAULT_MAX_CONCURRENT_TRADES,
                           feature_cols, 
                           target_cols,
                           DEFAULT_STOP_LOSS_ATR_MULTIPLIER,
                           DEFAULT_ATR_PREDICTED_WEIGHT,
                           model_params)
    
    # If no trades were made, return a very low score
    if metrics['trade_count'] == 0:
        return 100.0  # Return a high drawdown value for no trades
    
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
    normalized_sharpe = metrics['sharpe_ratio']  # Assuming max Sharpe around 2.0
    normalized_return = metrics['total_return'] / 100.0  # Convert percentage to decimal
    normalized_drawdown = abs(metrics['max_drawdown']) / 100.0  # Convert percentage to decimal
    
    # Calculate weights for each objective
    sharpe_weight = 0.4  # 40% weight to Sharpe ratio
    return_weight = 0.4  # 40% weight to total return
    drawdown_weight = 0.2  # 20% weight to minimizing drawdown
    
    composite_score = normalized_return #-normalized_drawdown
    
    # Save top parameters after each trial
    save_top_parameters(study, symbol)
    
    return composite_score


if __name__ == "__main__":
    # Load and prepare data
    data_path = f"DB/{symbol}_fifteenminute_indicators.csv"
    df, feature_cols, target_cols = preprocess_data(pd.read_csv(data_path))
    
    #Filter data by time range
    start_date = '2025-01-15'  # Format: 'YYYY-MM-DD'
    end_date = '2025-03-31'    # Format: 'YYYY-MM-DD'

    #Every bar is 15 minutes, so 5000 bars is 1250 hours, so 50 days
    #Calculate number of days from window_size
    buffered_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=DEFAULT_WINDOW_SIZE*15/60/24)
    df = df[(df['Date'] >= buffered_start_date) & (df['Date'] <= end_date)]
    
    # Run optimization or use default parameters
    if SKIP_OPTIMIZATION:
        print("\nUsing default trading parameters and default model parameters (optimization skipped)")
        print(f"Trading Parameters:")
        print(f"  Min Risk: {DEFAULT_MIN_RISK:.3f}")
        print(f"  Max Risk: {DEFAULT_MAX_RISK:.3f}")
        print(f"  Risk Scaling: {DEFAULT_SCALING}")
        print(f"  Risk:Reward: {DEFAULT_RR}")
        print(f"  Min Predicted Move: {DEFAULT_MIN_PREDICTED_MOVE:.3f}")
        print(f"  Partial Take Profit: {DEFAULT_PARTIAL_TAKE_PROFIT:.3f}")
        print(f"  Min Holding Period: {DEFAULT_MIN_HOLDING_PERIOD}")
        print(f"  Max Holding Period: {DEFAULT_MAX_HOLDING_PERIOD}")
        print(f"  Max Concurrent Trades: {DEFAULT_MAX_CONCURRENT_TRADES}")
        print(f"  Window Size: {DEFAULT_WINDOW_SIZE}")
        print(f"  Retrain Interval: {DEFAULT_RETREIN_INTERVAL}")
        print(f"  Stop Loss ATR Multiplier: {DEFAULT_STOP_LOSS_ATR_MULTIPLIER}")
        print(f"  ATR vs Predicted Weight: {DEFAULT_ATR_PREDICTED_WEIGHT}")
        print(f"  Maker Fee (Buy): {MAKER_FEE*100:.1f}%")
        print(f"  Taker Fee (Sell): {TAKER_FEE*100:.1f}%")
        print(f"  Total Round-trip Cost: {(MAKER_FEE + TAKER_FEE)*100:.1f}%")
        print("  Stop loss and take profit levels are adjusted to compensate for fees")
        print(f"Model Parameters:")
        print(f"  Learning Rate: 0.1000")
        print(f"  N Estimators: 50")
        print(f"  Max Depth: 4")
        print(f"  Min Child Weight: 10.00")
        print(f"  Subsample: 0.700")
        print(f"  Colsample by Tree: 0.700")
        print(f"  Reg Alpha: 0.50")
        print(f"  Reg Lambda: 1.00")
        best_metrics = run_strategy(df, 
                                    DEFAULT_MIN_RISK,
                                    DEFAULT_MAX_RISK, 
                                    DEFAULT_SCALING, 
                                    DEFAULT_RR, 
                                    DEFAULT_MIN_PREDICTED_MOVE, 
                                    DEFAULT_WINDOW_SIZE, 
                                    DEFAULT_RETREIN_INTERVAL,
                                    DEFAULT_PARTIAL_TAKE_PROFIT,
                                    DEFAULT_MIN_HOLDING_PERIOD,
                                    DEFAULT_MAX_HOLDING_PERIOD,
                                    DEFAULT_MAX_CONCURRENT_TRADES,
                                    feature_cols,
                                    target_cols,
                                    DEFAULT_STOP_LOSS_ATR_MULTIPLIER,
                                    DEFAULT_ATR_PREDICTED_WEIGHT,
                                    None)  # Use default model parameters
        
        # Calculate composite score
        composite_score = (
            best_metrics['total_return'] / 
            (abs(best_metrics['max_drawdown']) + 1e-8) * 
            np.sqrt(best_metrics['win_rate'] / 100)
        )
        
        # Default model parameters
        default_model_params = {
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
        
        results = [{
            'learning_rate': default_model_params['learning_rate'],
            'n_estimators': default_model_params['n_estimators'],
            'max_depth': default_model_params['max_depth'],
            'max_leaves': 0,  # Default value
            'min_child_weight': default_model_params['min_child_weight'],
            'gamma': 0.0,  # Default value
            'subsample': default_model_params['subsample'],
            'colsample_bytree': default_model_params['colsample_bytree'],
            'colsample_bylevel': 1.0,  # Default value
            'reg_lambda': default_model_params['reg_lambda'],
            'reg_alpha': default_model_params['reg_alpha'],
            'max_bin': 256,  # Default value
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
        print("\nRunning Optuna optimization for XGBoost model parameters...")
        
        # Create storage for the study
        storage_name = f"sqlite:///DB/{symbol}_model_study.db"
        
        # Handle study resumption
        study_name = f"{symbol}_model_optimization"
        if RESUME_STUDY is None:
            # Check if study exists
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_name)
                print(f"\nFound existing study with {len(study.trials)} trials. Resuming...")
            except:
                print("\nNo existing study found. Starting new study...")
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_name,
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
                )
        elif RESUME_STUDY:
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_name)
                print(f"\nResuming existing study with {len(study.trials)} trials...")
            except:
                print("\nNo existing study found. Starting new study...")
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_name,
                    direction='maximize',
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
                )
        else:
            print("\nStarting new study...")
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
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
        
        # Create model parameters from best_params
        best_model_params = {
            'learning_rate': best_params['learning_rate'],
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
        
        best_metrics = run_strategy(
            df,
            DEFAULT_MIN_RISK,
            DEFAULT_MAX_RISK,
            DEFAULT_SCALING,
            DEFAULT_RR,
            DEFAULT_MIN_PREDICTED_MOVE,
            DEFAULT_WINDOW_SIZE,
            DEFAULT_RETREIN_INTERVAL,
            DEFAULT_PARTIAL_TAKE_PROFIT,
            DEFAULT_MIN_HOLDING_PERIOD,
            DEFAULT_MAX_HOLDING_PERIOD,
            DEFAULT_MAX_CONCURRENT_TRADES,
            feature_cols,
            target_cols,
            DEFAULT_STOP_LOSS_ATR_MULTIPLIER,
            best_model_params
        )
        
        # Calculate final composite score
        final_composite_score = (
            (0.4 * best_metrics['sharpe_ratio'] / 2.0) +
            (0.4 * best_metrics['total_return'] / 100.0) -
            (0.2 * abs(best_metrics['max_drawdown']) / 100.0)
        )
        
        results = [{
            'learning_rate': best_params['learning_rate'],
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
            'total_return': best_metrics['total_return'],
            'final_capital': best_metrics['final_capital'],
            'sharpe_ratio': best_metrics['sharpe_ratio'],
            'win_rate': best_metrics['win_rate'],
            'max_drawdown': best_metrics['max_drawdown'],
            'cagr': best_metrics['cagr'],
            'mar': best_metrics['mar'],
            'trade_count': best_metrics['trade_count'],
            'composite_score': final_composite_score,
            'equity_curve': best_metrics['equity_curve'],
            'trade_history': best_metrics['trade_history']
        }]

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Get best parameters
    best_params = results_df.iloc[0]
    print("\nBest Model Parameters:")
    print(f"Learning Rate: {best_params['learning_rate']:.4f}")
    print(f"N Estimators: {best_params['n_estimators']}")
    print(f"Max Depth: {best_params['max_depth']}")
    print(f"Max Leaves: {best_params['max_leaves']}")
    print(f"Min Child Weight: {best_params['min_child_weight']:.2f}")
    print(f"Gamma: {best_params['gamma']:.2f}")
    print(f"Subsample: {best_params['subsample']:.3f}")
    print(f"Colsample by Tree: {best_params['colsample_bytree']:.3f}")
    print(f"Colsample by Level: {best_params['colsample_bylevel']:.3f}")
    print(f"Reg Lambda: {best_params['reg_lambda']:.2f}")
    print(f"Reg Alpha: {best_params['reg_alpha']:.2f}")
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
    plt.savefig('DB/charts/best_model_equity_curve_phases.png')
    print("\nPhase-wise equity curves saved to DB/charts/best_model_equity_curve_phases.png")

    # Also save the original full equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(best_params['equity_curve'].index, best_params['equity_curve'].values)
    plt.title('Best Model Parameters Equity Curve (Full Timeline)')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    # Make x-axis dates smaller to prevent overlapping
    plt.xticks(rotation=45, fontsize=8)
    plt.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    plt.savefig('DB/charts/best_model_equity_curve_full.png')
    print("Full equity curve saved to DB/charts/best_model_equity_curve_full.png")

    # Save results
    results_df.to_csv('DB/model_optimization_results.csv', index=False)
    best_params['trade_history'].to_csv('DB/best_model_trade_history.csv', index=False)
    print("\nResults saved to DB/model_optimization_results.csv")
    print("Best trade history saved to DB/best_model_trade_history.csv")

