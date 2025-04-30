import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import os
from tqdm import tqdm
import shap # Import SHAP
import json

# Load the data
symbol = "TSLA"
input_dataset = f"DB/{symbol}_15min_indicators.csv"
df = pd.read_csv(input_dataset)
df = df.iloc[:int(len(df) * 0.5)]
# Convert the index column to datetime if it exists
if 'Unnamed: 0' in df.columns:
    df['Date'] = pd.to_datetime(df['Unnamed: 0'])
    df = df.drop('Unnamed: 0', axis=1)
elif 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Sort by date to ensure proper time series handling
df = df.sort_values('Date')

# Create train and holdout sets (75% train, 25% holdout)
train_size = int(len(df) * 0.75)
train_df = df.iloc[:train_size].copy()
holdout_df = df.iloc[train_size:].copy()

print(f"\nData split:")
print(f"Training set size: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
print(f"Holdout set size: {len(holdout_df)} samples ({holdout_df.index.min()} to {holdout_df.index.max()})")

# Store original price for backtesting
train_df['Close_Original'] = train_df['Close'].copy()
holdout_df['Close_Original'] = holdout_df['Close'].copy()

print("\nStarting data preprocessing...")

# Identify column types (using only training data)
datetime_columns = train_df.select_dtypes(include=['datetime64']).columns
numerical_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
boolean_columns = train_df.select_dtypes(include=['bool']).columns
categorical_columns = train_df.select_dtypes(include=['object']).columns

print(f"\nFound {len(numerical_columns)} numerical columns")
print(f"Found {len(categorical_columns)} categorical columns")
print(f"Found {len(datetime_columns)} datetime columns")

# Remove Close_Original and datetime columns from numerical columns to avoid scaling them
numerical_columns = numerical_columns.drop('Close_Original')
numerical_columns = numerical_columns.drop(datetime_columns, errors='ignore')

print("\nHandling missing values...")
# Handle missing values using appropriate methods for each data type (separately for train and holdout)
for dataset in [train_df, holdout_df]:
    for col in dataset.columns:
        if col in numerical_columns:
            # For numerical columns, use expanding mean for first 100 records, then rolling mean
            dataset[col] = dataset[col].fillna(
                dataset[col].expanding(min_periods=1).mean().shift(1)
            ).fillna(
                dataset[col].rolling(window=100, min_periods=1).mean().shift(1)
            )
        elif col in categorical_columns:
            # For categorical columns, use forward fill with backward fill as backup
            dataset[col] = dataset[col].ffill().bfill()
        elif col in datetime_columns:
            # For datetime columns, use forward fill with backward fill as backup
            dataset[col] = dataset[col].ffill().bfill()
        else:
            # For any other columns, use forward fill with backward fill as backup
            dataset[col] = dataset[col].ffill().bfill()

print("Converting boolean features to numerical...")
# Convert boolean features to numerical
if len(boolean_columns) > 0:
    train_df[boolean_columns] = train_df[boolean_columns].astype(int)
    holdout_df[boolean_columns] = holdout_df[boolean_columns].astype(int)

print("Handling infinite values...")
# Handle infinite values in numerical columns only
for dataset in [train_df, holdout_df]:
    dataset[numerical_columns] = dataset[numerical_columns].replace([np.inf, -np.inf], np.nan)
    dataset[numerical_columns] = dataset[numerical_columns].fillna(
        dataset[numerical_columns].expanding(min_periods=1).mean().shift(1)
    ).fillna(
        dataset[numerical_columns].rolling(window=100, min_periods=1).mean().shift(1)
    )

print("Encoding categorical variables...")
# Encode categorical variables (fit on training data only)
label_encoders = {}
for col in categorical_columns:
    if col in train_df.columns:
        # Create a combined set of unique values from both train and holdout
        unique_values = pd.concat([
            train_df[col].astype(str),
            holdout_df[col].astype(str)
        ]).unique()
        
        # Create and fit the encoder with all possible values
        label_encoders[col] = LabelEncoder()
        label_encoders[col].fit(unique_values)
        
        # Transform both datasets using the same encoder
        train_df[col] = label_encoders[col].transform(train_df[col].astype(str))
        holdout_df[col] = label_encoders[col].transform(holdout_df[col].astype(str))

print("\nPreparing features and target...")
# Prepare features and target
target_cols = ['Price_Change_5']
exclude_cols = target_cols + list(categorical_columns) + list(datetime_columns)

# Add base OHLC columns to the exclude list
base_ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Close_Original', 'Volume']
exclude_cols.extend(base_ohlc_cols)

# Remove Regime_Label from exclude_cols if present
if 'Regime_Label' in exclude_cols:
    exclude_cols.remove('Regime_Label')

# Also remove regime probability columns from exclude_cols if present
regime_prob_cols = [col for col in train_df.columns if col.startswith('Regime_Prob_')]
for col in regime_prob_cols:
    if col in exclude_cols:
        exclude_cols.remove(col)

# Add Regime_Label and regime probability columns to feature_cols if present in the dataframe
feature_cols = [col for col in train_df.columns if col not in exclude_cols]
if 'Regime_Label' in train_df.columns and 'Regime_Label' not in feature_cols:
    feature_cols.append('Regime_Label')
for col in regime_prob_cols:
    if col not in feature_cols:
        feature_cols.append(col)

print(f"Total number of features: {len(feature_cols)}")

# Feature Selection Process
print("\nStarting feature selection process...")

# Define model parameters for feature selection
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
    'device': 'cuda'
}

# Implement walk-forward feature selection
print("\nImplementing walk-forward feature selection...")
# Use expanding windows for feature selection
initial_train_size = len(train_df) // 3  # Use first 33% of training data for initial training
feature_importance_scores = {}

# Calculate feature importance using expanding window
print("Calculating feature importance...")
# Process in batches to better utilize GPU
batch_size = 1000  # Process 1000 steps at a time
for batch_start in tqdm(range(initial_train_size, len(train_df), batch_size), desc="Feature Selection Progress"):
    batch_end = min(batch_start + batch_size, len(train_df))
    
    # Get training data for the entire batch
    train_data = train_df.iloc[:batch_end]
    
    # Scale features using rolling statistics for the entire batch
    X_train = train_data[feature_cols].copy()
    for col in feature_cols:
        # Use expanding mean/std
        mean = X_train[col].expanding(min_periods=1).mean().shift(1)
        std = X_train[col].expanding(min_periods=1).std().shift(1)
        
        # Standardize
        X_train[col] = (X_train[col] - mean) / (std + 1e-8)
    
    # Convert to numpy arrays and then to GPU format
    X_train_scaled = np.array(X_train.values, dtype=np.float32)
    y_train = np.array(train_data[target_cols].values.ravel(), dtype=np.float32)
    
    # Train model for feature importance
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train_scaled, y_train)
    
    # Get feature importance for this batch
    for feature, importance in zip(feature_cols, model.feature_importances_):
        if feature not in feature_importance_scores:
            feature_importance_scores[feature] = []
        feature_importance_scores[feature].append(importance)

# Calculate average importance scores
avg_importance = {feature: np.mean(scores) for feature, scores in feature_importance_scores.items()}
feature_importance = pd.DataFrame({
    'Feature': list(avg_importance.keys()),
    'Importance': list(avg_importance.values())
}).sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.barh(feature_importance['Feature'][:20], feature_importance['Importance'][:20])
plt.title('Top 20 Most Important Features (Expanding Window Analysis)')
plt.xlabel('Average Importance Score')
plt.tight_layout()
plt.savefig("DB/charts/feature_importance.png")
print("Feature importance chart saved to DB/charts/feature_importance.png")

# Select top features based on average importance
importance_threshold = 0.01  # Keep features with importance > 1%
selected_features = feature_importance[feature_importance['Importance'] > importance_threshold]['Feature'].tolist()
print(f"\nSelected {len(selected_features)} features based on expanding window importance analysis")

# Update feature_cols with final selected features
feature_cols = selected_features

# Save selected features and their importance scores to JSON
os.makedirs('Features', exist_ok=True)
features_file = f'Features/{symbol}.json'
features_to_save = {
    feature: float(importance)  # Convert numpy.float64 to Python float for JSON serialization
    for feature, importance in zip(feature_importance['Feature'], feature_importance['Importance'])
    if importance > importance_threshold  # Only include features above threshold
}

with open(features_file, 'w') as f:
    json.dump(features_to_save, f, indent=4)
print(f"\nFeatures and importance scores saved to {features_file}")
print(f"Saved {len(features_to_save)} features with importance > {importance_threshold}")

# Implement true walk-forward testing with expanding windows
print("\nProceeding with walk-forward model testing...")
walk_forward_metrics = []
predictions = []
actuals = []

# Process in batches to better utilize GPU
batch_size = 1000  # Process 1000 steps at a time
for batch_start in tqdm(range(initial_train_size, len(train_df), batch_size), desc="Model Testing Progress"):
    batch_end = min(batch_start + batch_size, len(train_df))
    
    # Get training data up to batch_start
    train_data = train_df.iloc[:batch_start]
    # Get validation data for the batch
    val_data = train_df.iloc[batch_start:batch_end]
    
    # Scale features using rolling statistics
    X_train = train_data[feature_cols].copy()
    X_val = val_data[feature_cols].copy()
    
    # Pre-calculate means and stds for the entire batch
    for col in feature_cols:
        # Use expanding mean/std for training
        mean = X_train[col].expanding(min_periods=1).mean().shift(1)
        std = X_train[col].expanding(min_periods=1).std().shift(1)
        
        # Standardize training data
        X_train[col] = (X_train[col] - mean) / (std + 1e-8)
        
        # Use the last mean and std from training for validation
        last_mean = mean.iloc[-1]
        last_std = std.iloc[-1]
        X_val[col] = (X_val[col] - last_mean) / (last_std + 1e-8)
    
    # Convert to numpy arrays and then to GPU format
    X_train_scaled = np.array(X_train.values, dtype=np.float32)
    X_val_scaled = np.array(X_val.values, dtype=np.float32)
    y_train = np.array(train_data[target_cols].values.ravel(), dtype=np.float32)
    y_val = np.array(val_data[target_cols].values.ravel(), dtype=np.float32)
    
    # Train model
    model = xgb.XGBRegressor(**model_params)
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    # Make predictions for the entire batch
    y_pred = model.predict(X_val_scaled)
    
    # Store predictions and actual values
    predictions.extend(y_pred)
    actuals.extend(y_val)
    
    # Calculate metrics for this batch
    mse = mean_squared_error(y_val, y_pred)
    mse_percentage = mse * 100
    r2 = r2_score(y_val, y_pred)
    directional_accuracy = np.mean((y_val > 0) == (y_pred > 0))
    
    walk_forward_metrics.append({
        'Step': batch_start,
        'MSE': mse_percentage,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    })

# Calculate final metrics
final_mse = mean_squared_error(actuals, predictions)
final_mse_percentage = final_mse * 100
final_r2 = r2_score(actuals, predictions)
final_directional_accuracy = np.mean((np.array(actuals) > 0) == (np.array(predictions) > 0))

print("\nFinal Performance Metrics:")
print(f"Mean Squared Error: {final_mse_percentage:.4f}%")
print(f"R² Score: {final_r2:.4f}")
print(f"Directional Accuracy: {final_directional_accuracy:.4f}")

# Create price prediction analysis chart
print("\nCreating price prediction analysis chart...")
plt.figure(figsize=(15, 8))

# Plot actual vs predicted price changes
plt.subplot(2, 1, 1)
plt.plot(actuals, label='Actual Price Change', alpha=0.7)
plt.plot(predictions, label='Predicted Price Change', alpha=0.7)
plt.title('Actual vs Predicted Price Changes')
plt.xlabel('Time')
plt.ylabel('Price Change')
plt.legend()
plt.grid(True)

# Plot prediction error
plt.subplot(2, 1, 2)
error = np.array(actuals) - np.array(predictions)
plt.plot(error, label='Prediction Error', color='red', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.title('Prediction Error Over Time')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("DB/charts/price_prediction_analysis.png")
print("Price prediction analysis chart saved to DB/charts/price_prediction_analysis.png")

print("\nTraining final model on all training data...")
# Need the final training data for SHAP
final_X_train = train_df.iloc[:initial_train_size + len(predictions)][feature_cols].copy()
final_y_train = train_df.iloc[:initial_train_size + len(predictions)][target_cols].copy()

# Scale the final training data using rolling statistics
for col in feature_cols:
    # Use expanding mean/std
    mean = final_X_train[col].expanding(min_periods=1).mean().shift(1)
    std = final_X_train[col].expanding(min_periods=1).std().shift(1)
    
    # Standardize
    final_X_train[col] = (final_X_train[col] - mean) / (std + 1e-8)

final_y_train_scaled = final_y_train.values.ravel()

# Train final model on all available training data (up to the end of walk-forward)
final_model = xgb.XGBRegressor(**model_params)
final_model.fit(final_X_train, final_y_train_scaled)

print("\nCalculating SHAP values...")
# Create a SHAP explainer
explainer = shap.TreeExplainer(final_model)
# Calculate SHAP values for the scaled training data
shap_values = explainer.shap_values(final_X_train)

print("Generating SHAP summary plots...")
# Create SHAP summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_values, final_X_train, feature_names=feature_cols, show=False)
plt.title('SHAP Summary Plot (Beeswarm)')
plt.tight_layout()
plt.savefig("DB/charts/shap_summary_beeswarm.png")
plt.close() # Close the plot to free memory
print("SHAP beeswarm plot saved to DB/charts/shap_summary_beeswarm.png")

# Create SHAP summary plot (bar)
plt.figure()
shap.summary_plot(shap_values, final_X_train, feature_names=feature_cols, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar)')
plt.tight_layout()
plt.savefig("DB/charts/shap_summary_bar.png")
plt.close() # Close the plot
print("SHAP bar plot saved to DB/charts/shap_summary_bar.png")

print("\nSaving model...")
# Save the model
os.makedirs("DB/models", exist_ok=True)
joblib.dump(final_model, "DB/models/price_prediction_model.joblib")
print("\nModel saved to DB/models/")

print("\nEvaluating model with rolling retraining on holdout set...")
# Initialize lists to store predictions and actual values
holdout_predictions = []
holdout_actuals = []
rolling_metrics = []

# Define retraining interval
retrain_interval = 50

# Process holdout set in chunks
for i in tqdm(range(0, len(holdout_df), retrain_interval), desc="Rolling Evaluation Progress"):
    # Get the current chunk of holdout data
    current_holdout = holdout_df.iloc[i:i+retrain_interval]
    
    if len(current_holdout) == 0:
        break
    
    # Combine all previous training data with all previous holdout data
    current_train = pd.concat([
        train_df,
        holdout_df.iloc[:i]
    ])
    
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
    
    # Store predictions and actual values
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

# Convert predictions and actuals to numpy arrays
holdout_predictions = np.array(holdout_predictions)
holdout_actuals = np.array(holdout_actuals)

# Calculate final metrics
final_mse = mean_squared_error(holdout_actuals, holdout_predictions)
final_r2 = r2_score(holdout_actuals, holdout_predictions)
final_dir_acc = np.mean((holdout_actuals > 0) == (holdout_predictions > 0))

print("\nFinal Holdout Set Performance Metrics:")
print(f"Mean Squared Error: {final_mse * 100:.4f}%")
print(f"R² Score: {final_r2:.4f}")
print(f"Directional Accuracy: {final_dir_acc:.4f}")

# Create rolling metrics visualization
plt.figure(figsize=(15, 12))

# Plot MSE over time
plt.subplot(3, 1, 1)
plt.plot([m['End_Date'] for m in rolling_metrics], [m['MSE'] for m in rolling_metrics])
plt.title('Rolling MSE')
plt.xlabel('Date')
plt.ylabel('MSE (%)')
plt.grid(True)

# Plot R² over time
plt.subplot(3, 1, 2)
plt.plot([m['End_Date'] for m in rolling_metrics], [m['R2'] for m in rolling_metrics])
plt.title('Rolling R²')
plt.xlabel('Date')
plt.ylabel('R²')
plt.grid(True)

# Plot Directional Accuracy over time
plt.subplot(3, 1, 3)
plt.plot([m['End_Date'] for m in rolling_metrics], [m['Directional_Accuracy'] for m in rolling_metrics])
plt.title('Rolling Directional Accuracy')
plt.xlabel('Date')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig("DB/charts/rolling_metrics.png")
print("Rolling metrics chart saved to DB/charts/rolling_metrics.png")

# Create prediction analysis chart
plt.figure(figsize=(15, 8))

# Plot actual vs predicted price changes
plt.subplot(2, 1, 1)
plt.plot(holdout_df.index[:len(holdout_actuals)], holdout_actuals, label='Actual Price Change', alpha=0.7)
plt.plot(holdout_df.index[:len(holdout_predictions)], holdout_predictions, label='Predicted Price Change', alpha=0.7)
plt.title('Holdout Set: Actual vs Predicted Price Changes (Rolling Retrain)')
plt.xlabel('Time')
plt.ylabel('Price Change')
plt.legend()
plt.grid(True)

# Plot prediction error
plt.subplot(2, 1, 2)
holdout_error = holdout_actuals - holdout_predictions
plt.plot(holdout_df.index[:len(holdout_error)], holdout_error, label='Prediction Error', color='red', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.title('Holdout Set: Prediction Error Over Time (Rolling Retrain)')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("DB/charts/holdout_prediction_analysis_rolling.png")
print("Holdout set prediction analysis chart saved to DB/charts/holdout_prediction_analysis_rolling.png")

# Save rolling metrics to CSV for further analysis
rolling_metrics_df = pd.DataFrame(rolling_metrics)
rolling_metrics_df.to_csv("DB/rolling_metrics.csv", index=False)
print("Rolling metrics saved to DB/rolling_metrics.csv")

# Save the final model and scaler after rolling holdout evaluation
joblib.dump(current_model, "DB/models/price_prediction_model_final.joblib")
print("Final model and scaler saved for live trading.")
