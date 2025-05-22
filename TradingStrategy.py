import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
import optuna
import json
from utils import preprocess_data, calculate_feature_importance, clamp, calculate_sharpe_ratio, calculate_max_drawdown, calculate_distribution_metrics
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from Optimization.FAPT_Wasserstein import predict_optimal_parameters, get_top_market_weather_features

import scipy.stats


symbol = "TSLA"

# Default parameters (used when skip_optimization=True)
DEFAULT_MIN_RISK = 0.006489884911079899
DEFAULT_MAX_RISK = 0.009567915879945087
DEFAULT_SCALING = 1.6949727011806939
DEFAULT_RR = 1.6037213842177254
DEFAULT_MIN_PREDICTED_MOVE = 0.005113433706915217
DEFAULT_MIN_HOLDING_PERIOD = 8
DEFAULT_MAX_HOLDING_PERIOD = 9
DEFAULT_PARTIAL_TAKE_PROFIT = 0.8675868861032823
DEFAULT_MAX_CONCURRENT_TRADES = 3
DEFAULT_WINDOW_SIZE = 359
DEFAULT_RETREIN_INTERVAL = 155

# Fixed parameters
INITIAL_CAPITAL = 1000000
SLIPPAGE = 0.0005
TRANSACTION_FEE = 1.0


# Flag to skip optimization
SKIP_OPTIMIZATION = False  # Set to True to use default parameters
USE_FAPT = True
OPTIMIZATION_TRIALS = 350
RESUME_STUDY = True  # Set to True to resume from previous study, False to start new, None to check if exists

MIN_WINDOW = 300
MAX_WINDOW = 20000

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




def run_strategy(df_window, min_risk_percentage, max_risk_percentage, risk_scaling_factor, risk_reward_ratio, min_predicted_move, window_size, retrain_interval, partial_take_profit, min_holding_period, max_holding_period, max_concurrent_trades, feature_cols, target_cols):
    """Run trading strategy with given parameters"""
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
    
    # Initialize predicted change column
    df_window['Predicted_Change'] = np.nan
    
    # Simulate trading with model retraining every retrain_interval bars
    capital = INITIAL_CAPITAL
    open_trades = []
    trade_history = []
    holdout_predictions = []
    holdout_actuals = []
    rolling_metrics = []
    equity_curve = pd.Series(index=df_window['Date'], data=INITIAL_CAPITAL, dtype=np.float64)
    
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
        if total_data_processed >= window_size:
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
                    profit = (exit_price - trade['entry_price']) * trade['size'] - 2 * TRANSACTION_FEE
                    capital += profit
                    trade['exit_idx'] = idx
                    trade['exit_price'] = exit_price
                    trade['result'] = 'TP'
                    trade['profit'] = profit
                    trade_history.append(trade)
                    closed_trades.append(trade)
                elif low <= trade['stop_loss']:
                    exit_price = trade['stop_loss'] * (1 - SLIPPAGE)
                    profit = (exit_price - trade['entry_price']) * trade['size'] - 2 * TRANSACTION_FEE
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
                    tp_partial = projected_entry + partial_take_profit * (projected_tp - projected_entry)
                    if high >= tp_partial:
                        exit_price = tp_partial * (1 - SLIPPAGE)
                        profit = (exit_price - trade['entry_price']) * trade['size'] - 2 * TRANSACTION_FEE
                        capital += profit
                        trade['exit_idx'] = idx
                        trade['exit_price'] = exit_price
                        trade['result'] = 'Partial TP'
                        trade['profit'] = profit
                        trade_history.append(trade)
                        closed_trades.append(trade)
                    elif low <= projected_entry:
                        exit_price = projected_entry * (1 - SLIPPAGE)
                        profit = (exit_price - trade['entry_price']) * trade['size'] - 2 * TRANSACTION_FEE
                        capital += profit
                        trade['exit_idx'] = idx
                        trade['exit_price'] = exit_price
                        trade['result'] = 'BE'
                        trade['profit'] = profit
                        trade_history.append(trade)
                        closed_trades.append(trade)
                
                if holding_period >= max_holding_period and trade not in closed_trades:
                    exit_price = row['Close'] * (1 - SLIPPAGE)
                    profit = (exit_price - trade['entry_price']) * trade['size'] - 2 * TRANSACTION_FEE
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
                    
                    risk_amount = capital * risk_percentage
                    stop_loss = entry_price * (1 - (predicted_move / risk_reward_ratio))
                    take_profit = entry_price * (1 + predicted_move)
                    risk_per_share = entry_price - stop_loss
                    
                    if risk_per_share <= 0 or np.isnan(risk_per_share):
                        continue
                        
                    size = risk_amount / risk_per_share
                    size = min(size, capital / entry_price)
                    size = np.floor(size)
                    
                    if size <= 0:
                        continue
                        
                    open_trades.append({
                        'entry_idx': idx,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': size,
                        'capital_at_entry': capital,
                        'predicted_change': row['Predicted_Change'],
                        'risk_percentage': risk_percentage,
                        'result': None,
                        'exit_idx': None,
                        'exit_price': None,
                        'profit': None
                    })
            
            # Update equity curve
            equity_curve.iloc[idx] = np.float64(capital)
    
    # Close any remaining open trades
    for trade in open_trades:
        exit_price = df_window.iloc[-1]['Close'] * (1 - SLIPPAGE)
        profit = (exit_price - trade['entry_price']) * trade['size'] - 2 * TRANSACTION_FEE
        trade['exit_idx'] = len(df_window) - 1
        trade['exit_price'] = exit_price
        trade['result'] = 'EOD'
        trade['profit'] = profit
        capital += profit
        trade_history.append(trade)
    
    # Calculate performance metrics
    results_df = pd.DataFrame(trade_history)
    returns = equity_curve.pct_change().dropna()
    
    # Handle case where no trades were made
    if len(results_df) == 0:
        metrics = {
            'total_return': 0.0,
            'final_capital': capital,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'trade_count': 0,
            'equity_curve': equity_curve,
            'trade_history': results_df
        }
    else:
        metrics = {
            'total_return': (capital / INITIAL_CAPITAL - 1) * 100,
            'final_capital': capital,
            'sharpe_ratio': calculate_sharpe_ratio(returns),
            'win_rate': np.mean(results_df['profit'] > 0) * 100,
            'max_drawdown': calculate_max_drawdown(equity_curve) * 100,
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
        
        # Get the equity curve from trial attributes and convert back to Series
        equity_curve_dict = attrs.get('equity_curve')
        if equity_curve_dict is not None:
            equity_curve = pd.Series(
                data=equity_curve_dict['values'],
                index=pd.to_datetime(equity_curve_dict['dates'])
            )
            
            # Plot and save equity curve phases for this trial
            plt.figure(figsize=(15, 10))
            
            # Split the timeline into 4 phases
            total_points = len(equity_curve)
            phase_size = total_points // 4
            
            # Create subplots for each phase
            for j in range(4):
                start_idx = j * phase_size
                end_idx = (j + 1) * phase_size if j < 3 else total_points
                
                plt.subplot(2, 2, j + 1)
                phase_data = equity_curve.iloc[start_idx:end_idx]
                plt.plot(phase_data.index, phase_data.values)
                plt.title(f'Phase {j + 1} Equity Curve')
                plt.xlabel('Date')
                plt.ylabel('Capital ($)')
                plt.grid(True)
                
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
            'parameters': {
                'min_risk_percentage': params['min_risk_percentage'],
                'max_risk_percentage': params['max_risk_percentage'],
                'risk_scaling_factor': params['risk_scaling_factor'],
                'risk_reward_ratio': params['risk_reward_ratio'],
                'min_predicted_move': params['min_predicted_move'],
                'partial_take_profit': params['partial_take_profit'],
                'min_holding_period': params['min_holding_period'],
                'max_holding_period': params['max_holding_period'],
                'max_concurrent_trades': params['max_concurrent_trades'],
                'window_fraction': params['window_fraction'],
                'retrain_fraction': params['retrain_fraction'],
                'calculated_window_size': window_size,
                'calculated_retrain_interval': retrain_interval
            },
            'metrics': {
                'total_return': attrs.get('total_return', 0.0),
                'sharpe_ratio': attrs.get('sharpe_ratio', 0.0),
                'win_rate': attrs.get('win_rate', 0.0),
                'max_drawdown': attrs.get('max_drawdown', 0.0),
                'trade_count': attrs.get('trade_count', 0)
            }
        })
    
    # Save to JSON
    os.makedirs('Parameters', exist_ok=True)
    params_file = f'Parameters/{symbol}_Full_Optimization.json'
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=4)
    print(f"\nTop 10 parameters saved to {params_file}")
    print(f"Top 10 equity curve phase graphs saved to Visualization folder")

def objective(trial):
    """Objective function for Optuna optimization"""
    # Define parameter ranges
    min_risk = trial.suggest_float('min_risk_percentage', 0.005, 0.02)
    max_risk = trial.suggest_float('max_risk_percentage', min_risk, min(0.05, min_risk*1.5))
    scaling_factor = trial.suggest_float('risk_scaling_factor', 1.5, 3.0)
    reward_ratio = trial.suggest_float('risk_reward_ratio', 1.5, 3.0)
    min_predicted_move = trial.suggest_float('min_predicted_move', 0.005, 0.01)
    window_fraction = trial.suggest_float('window_fraction', 0.01, 0.5) # 1% to 50% of the data
    retrain_fraction = trial.suggest_float('retrain_fraction', 0.05, 1) # 5% to 100% of the window size
    window_size = clamp(int(len(df) * window_fraction), MIN_WINDOW, MAX_WINDOW)
    retrain_interval = max(int(window_size * retrain_fraction), 10)
    partial_take_profit = trial.suggest_float('partial_take_profit', 0.7, 0.95)
    min_holding_period = trial.suggest_int('min_holding_period', 5, 20)
    max_holding_period = trial.suggest_int('max_holding_period', min_holding_period, 40)
    max_concurrent_trades = trial.suggest_int('max_concurrent_trades', 1, 10)
    
    # Run strategy with current parameters
    metrics = run_strategy(df, 
                           min_risk, 
                           max_risk, 
                           scaling_factor, 
                           reward_ratio, 
                           min_predicted_move, 
                           window_size, 
                           retrain_interval, 
                           partial_take_profit, 
                           min_holding_period, 
                           max_holding_period, 
                           max_concurrent_trades,
                           feature_cols, 
                           target_cols)
    
    # If no trades were made, return a very low score
    if metrics['trade_count'] == 0:
        return -1000.0
    
    # Store metrics as trial attributes
    trial.set_user_attr('total_return', metrics['total_return'])
    trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
    trial.set_user_attr('win_rate', metrics['win_rate'])
    trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
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
    
    composite_score = -normalized_drawdown
    
    # Save top parameters after each trial
    save_top_parameters(study, symbol)
    
    return composite_score


if __name__ == "__main__":
    # Load and prepare data
    data_path = f"DB/{symbol}_15min_indicators.csv"
    df, feature_cols, target_cols = preprocess_data(pd.read_csv(data_path))
    
    # Filter data by time range
    # start_date = '2015-04-01'  # Format: 'YYYY-MM-DD'
    # end_date = '2019-03-01'    # Format: 'YYYY-MM-DD'
    # df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Run optimization or use default parameters
    if SKIP_OPTIMIZATION:
        print("\nUsing default parameters (optimization skipped)")
        print(f"Min Risk: {DEFAULT_MIN_RISK:.3f}")
        print(f"Max Risk: {DEFAULT_MAX_RISK:.3f}")
        print(f"Risk Scaling: {DEFAULT_SCALING}")
        print(f"Risk:Reward: {DEFAULT_RR}")
        print(f"Min Predicted Move: {DEFAULT_MIN_PREDICTED_MOVE:.3f}")
        print(f"Partial Take Profit: {DEFAULT_PARTIAL_TAKE_PROFIT:.3f}")
        print(f"Min Holding Period: {DEFAULT_MIN_HOLDING_PERIOD}")
        print(f"Max Holding Period: {DEFAULT_MAX_HOLDING_PERIOD}")
        print(f"Max Concurrent Trades: {DEFAULT_MAX_CONCURRENT_TRADES}")
        print(f"Window Size: {DEFAULT_WINDOW_SIZE}")
        print(f"Retrain Interval: {DEFAULT_RETREIN_INTERVAL}")
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
                                    target_cols)
        
        # Calculate composite score
        composite_score = (
            best_metrics['total_return'] / 
            (abs(best_metrics['max_drawdown']) + 1e-8) * 
            np.sqrt(best_metrics['win_rate'] / 100)
        )
        
        results = [{
            'min_risk_percentage': DEFAULT_MIN_RISK,
            'max_risk_percentage': DEFAULT_MAX_RISK,
            'risk_scaling_factor': DEFAULT_SCALING,
            'risk_reward_ratio': DEFAULT_RR,
            'min_predicted_move': DEFAULT_MIN_PREDICTED_MOVE,
            'window_size': DEFAULT_WINDOW_SIZE,
            'retrain_interval': DEFAULT_RETREIN_INTERVAL,
            'partial_take_profit': DEFAULT_PARTIAL_TAKE_PROFIT,
            'min_holding_period': DEFAULT_MIN_HOLDING_PERIOD,
            'max_holding_period': DEFAULT_MAX_HOLDING_PERIOD,
            'max_concurrent_trades': DEFAULT_MAX_CONCURRENT_TRADES,
            'total_return': best_metrics['total_return'],
            'final_capital': best_metrics['final_capital'],
            'sharpe_ratio': best_metrics['sharpe_ratio'],
            'win_rate': best_metrics['win_rate'],
            'max_drawdown': best_metrics['max_drawdown'],
            'trade_count': best_metrics['trade_count'],
            'composite_score': composite_score,
            'equity_curve': best_metrics['equity_curve'],
            'trade_history': best_metrics['trade_history']
        }]
    else:
        print("\nRunning Optuna optimization...")
        
        # Create storage for the study
        storage_name = f"sqlite:///DB/{symbol}_study.db"
        
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
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
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
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
                )
        else:
            print("\nStarting new study...")
            study = optuna.create_study(
                study_name=symbol,
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
        # Calculate actual window_size and retrain_interval from fractions
        window_size = clamp(int(len(df) * best_params['window_fraction']), MIN_WINDOW, MAX_WINDOW)
        retrain_interval = max(int(window_size * best_params['retrain_fraction']), 10)
        
        best_metrics = run_strategy(
            df,
            best_params['min_risk_percentage'],
            best_params['max_risk_percentage'],
            best_params['risk_scaling_factor'],
            best_params['risk_reward_ratio'],
            best_params['min_predicted_move'],
            window_size,
            retrain_interval,
            best_params['partial_take_profit'],
            best_params['min_holding_period'],
            best_params['max_holding_period'],
            best_params['max_concurrent_trades'],
            feature_cols,
            target_cols
        )
        
        # Calculate final composite score
        final_composite_score = (
            (0.4 * best_metrics['sharpe_ratio'] / 2.0) +
            (0.4 * best_metrics['total_return'] / 100.0) -
            (0.2 * abs(best_metrics['max_drawdown']) / 100.0)
        )
        
        results = [{
            'min_risk_percentage': best_params['min_risk_percentage'],
            'max_risk_percentage': best_params['max_risk_percentage'],
            'risk_scaling_factor': best_params['risk_scaling_factor'],
            'risk_reward_ratio': best_params['risk_reward_ratio'],
            'min_predicted_move': best_params['min_predicted_move'],
            'window_fraction': best_params['window_fraction'],
            'retrain_fraction': best_params['retrain_fraction'],
            'window_size': window_size,
            'retrain_interval': retrain_interval,
            'partial_take_profit': best_params['partial_take_profit'],
            'min_holding_period': best_params['min_holding_period'],
            'max_holding_period': best_params['max_holding_period'],
            'max_concurrent_trades': best_params['max_concurrent_trades'],
            'total_return': best_metrics['total_return'],
            'final_capital': best_metrics['final_capital'],
            'sharpe_ratio': best_metrics['sharpe_ratio'],
            'win_rate': best_metrics['win_rate'],
            'max_drawdown': best_metrics['max_drawdown'],
            'trade_count': best_metrics['trade_count'],
            'composite_score': final_composite_score,
            'equity_curve': best_metrics['equity_curve'],
            'trade_history': best_metrics['trade_history']
        }]

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Get best parameters
    best_params = results_df.iloc[0]
    print("\nBest Parameters:")
    print(f"Min Risk: {best_params['min_risk_percentage']:.3f}")
    print(f"Max Risk: {best_params['max_risk_percentage']:.3f}")
    print(f"Risk Scaling: {best_params['risk_scaling_factor']:.2f}")
    print(f"Risk:Reward: {best_params['risk_reward_ratio']:.2f}")
    print(f"Min Predicted Move: {best_params['min_predicted_move']:.3f}")
    print(f"Window Size: {best_params['window_size']}")
    print(f"Retrain Interval: {best_params['retrain_interval']}")
    print(f"Partial Take Profit: {best_params['partial_take_profit']:.3f}")
    print(f"Min Holding Period: {best_params['min_holding_period']}")
    print(f"Max Holding Period: {best_params['max_holding_period']}")
    print(f"Max Concurrent Trades: {best_params['max_concurrent_trades']}")
    print(f"\nPerformance Metrics:")
    print(f"Total Return: {best_params['total_return']:.2f}%")
    print(f"Final Capital: ${best_params['final_capital']:,.2f}")
    print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    print(f"Win Rate: {best_params['win_rate']:.2f}%")
    print(f"Max Drawdown: {best_params['max_drawdown']:.2f}%")
    print(f"Trade Count: {best_params['trade_count']}")
    print(f"Composite Score: {best_params['composite_score']:.2f}")

    # Plot best equity curve
    plt.figure(figsize=(15, 10))

    # Split the timeline into 4 phases
    total_points = len(best_params['equity_curve'])
    phase_size = total_points // 4

    # Create subplots for each phase
    for i in range(4):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < 3 else total_points
        
        plt.subplot(2, 2, i + 1)
        phase_data = best_params['equity_curve'].iloc[start_idx:end_idx]
        plt.plot(phase_data.index, phase_data.values)
        plt.title(f'Phase {i + 1} Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.grid(True)
        
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
    plt.tight_layout()
    plt.savefig('DB/charts/best_equity_curve_full.png')
    print("Full equity curve saved to DB/charts/best_equity_curve_full.png")

    # Save results
    results_df.to_csv('DB/parameter_optimization_results.csv', index=False)
    best_params['trade_history'].to_csv('DB/best_parameter_trade_history.csv', index=False)
    print("\nResults saved to DB/parameter_optimization_results.csv")
    print("Best trade history saved to DB/best_parameter_trade_history.csv")

