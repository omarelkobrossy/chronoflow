import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
import optuna
import json
from utils import preprocess_data, calculate_feature_importance
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


symbol = "TSLA"

# Default parameters (used when skip_optimization=True)
DEFAULT_MAX_RISK = 0.036
DEFAULT_SCALING = 1.5
DEFAULT_RR = 1.5
DEFAULT_MIN_PREDICTED_MOVE = 0.009
DEFAULT_WINDOW_SIZE = 10000
DEFAULT_RETREIN_INTERVAL = 5000

# Fixed parameters
INITIAL_CAPITAL = 10000
BASE_RISK_PERCENTAGE = 0.015
MAX_CONCURRENT_TRADES = 5
SLIPPAGE = 0.0005
TRANSACTION_FEE = 1.0
MAX_HOLDING_PERIOD = 20
MIN_HOLDING_PERIOD = 5

# Flag to skip optimization
SKIP_OPTIMIZATION = True  # Set to True to use default parameters

# # Load model and scaler
# MODEL_PATH = "DB/models/price_prediction_model_final.joblib"
# SCALER_PATH = "DB/models/final_scaler_final.joblib"
# SELECTED_FEATURES_PATH = f"Features/{symbol}.json"

# assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
# assert os.path.exists(SCALER_PATH), f"Scaler not found at {SCALER_PATH}"
# assert os.path.exists(SELECTED_FEATURES_PATH), f"Selected features file not found at {SELECTED_FEATURES_PATH}"

# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# feature_cols = list(json.load(open(SELECTED_FEATURES_PATH)).keys())

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio from returns series"""
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-8)

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve"""
    rolling_max = equity_curve.expanding().max()
    drawdowns = equity_curve / rolling_max - 1
    return drawdowns.min()

def run_strategy(df, max_risk_percentage, risk_scaling_factor, risk_reward_ratio, min_predicted_move, window_size, retrain_interval):
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
        'device': 'cuda'  # Changed from 'cuda' to 'cpu' for consistency
    }
    # Initialize predicted change column
    df['Predicted_Change'] = np.nan
    
    # Simulate trading with model retraining every retrain_interval bars
    capital = INITIAL_CAPITAL
    open_trades = []
    trade_history = []
    # Initialize lists to store predictions and actual values
    holdout_predictions = []
    holdout_actuals = []
    rolling_metrics = []
    equity_curve = pd.Series(index=df['Date'], data=INITIAL_CAPITAL, dtype=np.float64)
    
    n = len(df)
    
    # Initialize model and feature importance for first window
    print("\nInitializing first window...")
    initial_window = df.iloc[:window_size].copy()
    initial_features = calculate_feature_importance(
        initial_window, 
        feature_cols, 
        target_cols,
        iterations=1,
        save_importance=False,
        visualize_importance=False
    )
    model = xgb.XGBRegressor(**model_params)
    
    # Scale initial window data
    X_initial = initial_window[initial_features].copy()
    for col in initial_features:
        # Use expanding mean/std for scaling with shift to prevent look-ahead bias
        mean = X_initial[col].expanding(min_periods=1).mean().shift(1)
        std = X_initial[col].expanding(min_periods=1).std().shift(1)
        X_initial[col] = (X_initial[col] - mean) / (std + 1e-8)
    
    y_initial = initial_window[target_cols].values.ravel()
    model.fit(X_initial, y_initial)
    
    # Process data in chunks
    print("\nProcessing data in chunks...")
    current_features = initial_features  # Start with initial features
    
    for start in range(window_size, n, retrain_interval):
        end = min(start + retrain_interval, n)
        # Only recalculate feature importance every 500 retrain intervals
        if (start - window_size) % (retrain_interval * 500) == 0:
            print(f"\nRecalculating feature importance at index {start}...")
            # Use only past data for feature selection
            feature_selection_data = df.iloc[start-window_size:start].copy()
            current_features = calculate_feature_importance(
                feature_selection_data,
                feature_cols,
                target_cols,
                iterations=1,
                save_importance=False,
                visualize_importance=False
            )
        current_holdout = df.iloc[start:end].copy()
        current_train = df.iloc[max(0, start - window_size):start].copy()

        # Prepare features for current training data
        current_X_train = current_train[feature_cols].copy()
        current_y_train = current_train[target_cols].values.ravel()
        
        # Prepare features for current holdout chunk
        current_X_holdout = current_holdout[feature_cols].copy()
        current_y_holdout = current_holdout[target_cols].values.ravel()
        
        # Scale features using rolling statistics
        for col in feature_cols:
            # Use expanding mean/std for training
            mean = current_X_train[col].expanding(min_periods=1).mean().shift(1)
            std = current_X_train[col].expanding(min_periods=1).std().shift(1)
            
            # Standardize training data
            current_X_train[col] = (current_X_train[col] - mean) / (std + 1e-8)
            
            # Use the last mean and std from training for validation
            last_mean = mean.iloc[-1]
            last_std = std.iloc[-1]
            current_X_holdout[col] = (current_X_holdout[col] - last_mean) / (last_std + 1e-8)
        
        # Train model on current training data
        current_model = xgb.XGBRegressor(**model_params)
        current_model.fit(current_X_train, current_y_train)

        # Make predictions on current holdout chunk
        chunk_predictions = current_model.predict(current_X_holdout)

        holdout_predictions.extend(chunk_predictions)
        holdout_actuals.extend(current_y_holdout)    

        # Calculate metrics for this chunk
        chunk_mse = mean_squared_error(current_y_holdout, chunk_predictions)
        chunk_r2 = r2_score(current_y_holdout, chunk_predictions)
        chunk_dir_acc = np.mean((current_y_holdout > 0) == (chunk_predictions > 0))
        
        rolling_metrics.append({
            'Start_Date': current_holdout.index[0],
            'End_Date': current_holdout.index[-1],
            'MSE': chunk_mse * 100,  # Convert to percentage
            'R2': chunk_r2,
            'Directional_Accuracy': chunk_dir_acc,
            'Training_Size': len(current_train)
        })
        print(rolling_metrics[-1])
        
        # Run trading for this window
        for idx in range(start, end):
            row = df.iloc[idx]
            print(row['Predicted_Change'])
            
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
                elif holding_period >= MIN_HOLDING_PERIOD:
                    projected_tp = trade['take_profit']
                    projected_entry = trade['entry_price']
                    tp_80 = projected_entry + 0.8 * (projected_tp - projected_entry)
                    if high >= tp_80:
                        exit_price = tp_80 * (1 - SLIPPAGE)
                        profit = (exit_price - trade['entry_price']) * trade['size'] - 2 * TRANSACTION_FEE
                        capital += profit
                        trade['exit_idx'] = idx
                        trade['exit_price'] = exit_price
                        trade['result'] = '80TP'
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
                
                if holding_period >= MAX_HOLDING_PERIOD and trade not in closed_trades:
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
            if len(open_trades) < MAX_CONCURRENT_TRADES:
                if row['Predicted_Change'] < -min_predicted_move:
                    entry_price = row['Open'] * (1 + SLIPPAGE)
                    
                    predicted_move = abs(row['Predicted_Change'])
                    risk_percentage = min(
                        BASE_RISK_PERCENTAGE * (1 + (predicted_move / min_predicted_move) * risk_scaling_factor),
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
        exit_price = df.iloc[-1]['Close'] * (1 - SLIPPAGE)
        profit = (exit_price - trade['entry_price']) * trade['size'] - 2 * TRANSACTION_FEE
        trade['exit_idx'] = len(df) - 1
        trade['exit_price'] = exit_price
        trade['result'] = 'EOD'
        trade['profit'] = profit
        capital += profit
        trade_history.append(trade)
    
    # Calculate performance metrics
    results_df = pd.DataFrame(trade_history)
    returns = equity_curve.pct_change().dropna()
    
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

def objective(trial):
    """Objective function for Optuna optimization"""
    # Define parameter ranges
    max_risk = trial.suggest_float('max_risk_percentage', 0.02, 0.045)
    scaling_factor = trial.suggest_float('risk_scaling_factor', 1.5, 3.0)
    reward_ratio = trial.suggest_float('risk_reward_ratio', 1.5, 3.0)
    min_predicted_move = trial.suggest_float('min_predicted_move', 0.005, 0.01)
    window_size = trial.suggest_int('window_size', 50, 15000)
    retrain_interval = trial.suggest_int('retrain_interval', 50, 5000)
    
    # Run strategy with current parameters
    metrics = run_strategy(df, max_risk, scaling_factor, reward_ratio, min_predicted_move, window_size, retrain_interval)
    
    # Store metrics as trial attributes
    trial.set_user_attr('total_return', metrics['total_return'])
    trial.set_user_attr('sharpe_ratio', metrics['sharpe_ratio'])
    trial.set_user_attr('win_rate', metrics['win_rate'])
    trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
    trial.set_user_attr('trade_count', metrics['trade_count'])
    
    # Calculate composite score that balances multiple objectives
    # 1. Normalize metrics to similar scales
    normalized_sharpe = metrics['sharpe_ratio'] / 2.0  # Assuming max Sharpe around 2.0
    normalized_return = metrics['total_return'] / 100.0  # Convert percentage to decimal
    normalized_drawdown = abs(metrics['max_drawdown']) / 100.0  # Convert percentage to decimal
    
    # 2. Calculate weights for each objective
    sharpe_weight = 0.4  # 40% weight to Sharpe ratio
    return_weight = 0.4  # 40% weight to total return
    drawdown_weight = 0.2  # 20% weight to minimizing drawdown
    
    # 3. Calculate composite score
    composite_score = (
        (sharpe_weight * normalized_sharpe) +
        (return_weight * normalized_return) -
        (drawdown_weight * normalized_drawdown)
    )
    
    return composite_score

# Load and prepare data
data_path = f"DB/{symbol}_15min_indicators.csv"
df, feature_cols, target_cols = preprocess_data(pd.read_csv(data_path))
# df = df.iloc[int(len(df) * 0.5):]
# if 'Unnamed: 0' in df.columns:
#     df['Date'] = pd.to_datetime(df['Unnamed: 0'])
#     df = df.drop('Unnamed: 0', axis=1)
# elif 'Date' in df.columns:
#     df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values('Date').reset_index(drop=True)
# df = df.dropna(subset=feature_cols)

# Run optimization or use default parameters
if SKIP_OPTIMIZATION:
    print("\nUsing default parameters (optimization skipped)")
    print(f"Max Risk: {DEFAULT_MAX_RISK:.3f}")
    print(f"Risk Scaling: {DEFAULT_SCALING}")
    print(f"Risk:Reward: {DEFAULT_RR}")
    print(f"Min Predicted Move: {DEFAULT_MIN_PREDICTED_MOVE:.3f}")
    
    best_metrics = run_strategy(df, 
                                DEFAULT_MAX_RISK, 
                                DEFAULT_SCALING, 
                                DEFAULT_RR, 
                                DEFAULT_MIN_PREDICTED_MOVE, 
                                DEFAULT_WINDOW_SIZE, 
                                DEFAULT_RETREIN_INTERVAL)
    
    # Calculate composite score
    composite_score = (
        best_metrics['total_return'] / 
        (abs(best_metrics['max_drawdown']) + 1e-8) * 
        np.sqrt(best_metrics['win_rate'] / 100)
    )
    
    results = [{
        'max_risk_percentage': DEFAULT_MAX_RISK,
        'risk_scaling_factor': DEFAULT_SCALING,
        'risk_reward_ratio': DEFAULT_RR,
        'min_predicted_move': DEFAULT_MIN_PREDICTED_MOVE,
        'window_size': DEFAULT_WINDOW_SIZE,
        'retrain_interval': DEFAULT_RETREIN_INTERVAL,
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
    # Create Optuna study with multi-objective optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_metrics = run_strategy(
        df,
        best_params['max_risk_percentage'],
        best_params['risk_scaling_factor'],
        best_params['risk_reward_ratio'],
        best_params['min_predicted_move'],
        best_params['window_size'],
        best_params['retrain_interval']
    )
    
    # Calculate final composite score
    final_composite_score = (
        (0.4 * best_metrics['sharpe_ratio'] / 2.0) +
        (0.4 * best_metrics['total_return'] / 100.0) -
        (0.2 * abs(best_metrics['max_drawdown']) / 100.0)
    )
    
    results = [{
        'max_risk_percentage': best_params['max_risk_percentage'],
        'risk_scaling_factor': best_params['risk_scaling_factor'],
        'risk_reward_ratio': best_params['risk_reward_ratio'],
        'min_predicted_move': best_params['min_predicted_move'],
        'window_size': best_params['window_size'],
        'retrain_interval': best_params['retrain_interval'],
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
print(f"Max Risk: {best_params['max_risk_percentage']:.3f}")
print(f"Risk Scaling: {best_params['risk_scaling_factor']:.2f}")
print(f"Risk:Reward: {best_params['risk_reward_ratio']:.2f}")
print(f"Min Predicted Move: {best_params['min_predicted_move']:.3f}")
print(f"Window Size: {best_params['window_size']}")
print(f"Retrain Interval: {best_params['retrain_interval']}")
print(f"\nPerformance Metrics:")
print(f"Total Return: {best_params['total_return']:.2f}%")
print(f"Final Capital: ${best_params['final_capital']:,.2f}")
print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
print(f"Win Rate: {best_params['win_rate']:.2f}%")
print(f"Max Drawdown: {best_params['max_drawdown']:.2f}%")
print(f"Trade Count: {best_params['trade_count']}")
print(f"Composite Score: {best_params['composite_score']:.2f}")

# Plot best equity curve
plt.figure(figsize=(12, 6))
plt.plot(best_params['equity_curve'].index, best_params['equity_curve'].values)
plt.title('Best Parameter Set Equity Curve')
plt.xlabel('Date')
plt.ylabel('Capital ($)')
plt.grid(True)
plt.tight_layout()
plt.savefig('DB/charts/best_equity_curve.png')

# Save results
results_df.to_csv('DB/parameter_optimization_results.csv', index=False)
best_params['trade_history'].to_csv('DB/best_parameter_trade_history.csv', index=False)
print("\nResults saved to DB/parameter_optimization_results.csv")
print("Best trade history saved to DB/best_parameter_trade_history.csv")

# Save model with symbol-specific name
# model_path = f"DB/models/{symbol}_LiveDeployModel.joblib"
# joblib.dump({
#     'model': model,
#     'scaler': scaler,
#     'feature_cols': feature_cols,
#     'parameters': best_params.to_dict(),
#     'metrics': best_metrics
# }, model_path)
# print(f"\nLive deployment model saved to {model_path}")

# Save optimized parameters to JSON if optimization was run
if not SKIP_OPTIMIZATION:
    # Create Parameters directory if it doesn't exist
    os.makedirs('Parameters', exist_ok=True)
    
    # Save parameters to JSON
    params_file = f'Parameters/{symbol}.json'
    params_to_save = {
        'max_risk_percentage': best_params['max_risk_percentage'],
        'risk_scaling_factor': best_params['risk_scaling_factor'],
        'risk_reward_ratio': best_params['risk_reward_ratio'],
        'min_predicted_move': best_params['min_predicted_move'],
        'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'performance_metrics': {
            'total_return': best_metrics['total_return'],
            'sharpe_ratio': best_metrics['sharpe_ratio'],
            'win_rate': best_metrics['win_rate'],
            'max_drawdown': best_metrics['max_drawdown'],
            'trade_count': best_metrics['trade_count']
        }
    }
    
    with open(params_file, 'w') as f:
        json.dump(params_to_save, f, indent=4)
    print(f"\nOptimized parameters saved to {params_file}")
