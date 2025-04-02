import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load the preprocessed data
input_dataset = "aapl_df_indicators_preprocessed.csv"
df = pd.read_csv(input_dataset)

# Identify categorical columns
categorical_columns = ['Date', 'Time', 'ORB_Status', 'Session', 'Price_Structure', 'Time_of_Day']

# Encode categorical variables
print("Encoding categorical variables...")
label_encoders = {}
for col in categorical_columns:
    if col in df.columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Prepare features and target
print("Preparing features...")
target_cols = ['Price_Change', 'Price_Change_5']
feature_cols = [
    # Price Action Features
    'Body_Size_MA5', 'Rolling_Std_10', 'Shadow_Balance',
    'Dynamic_Support', 'Price_Tick', 'Body_To_Range_Ratio',
    
    # Market Context Features
    'DIA_Corr_20', 'DIA_Beta_60',
    
    # Volume Features
    'Volume_Profile', 'Volume_Shift',
    
    # Composite Features
    'Vol_Body_Interaction', 'Price_Action_Signal',
    'Market_Impact', 'SR_Strength', 'Volatility_Regime',
    'Enhanced_Volume_Impact', 'Trend_Strength', 'SR_Break_Potential'
]

# Ensure all selected features exist in the dataframe
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols].copy()
y = df['Price_Change']

# Handle missing and infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Ensure all features are numeric
X = X.astype(float)
y = y.astype(float)

# Apply PCA to reduce multicollinearity
print("Applying PCA...")
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X)
print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} features while retaining 95% variance")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train XGBoost model with optimized parameters
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Transform feature importances back to original features
print("\nFeature Importance Analysis:")
pca_importance = pd.Series(model.feature_importances_)
original_importance = pd.Series(
    np.abs(pca.components_.T.dot(pca_importance)),
    index=feature_cols
)
original_importance = original_importance.sort_values(ascending=False)

# Plot transformed feature importances
plt.figure(figsize=(12, 6))
original_importance.plot(kind='bar')
plt.title('Feature Importance (Including Composite Features)')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('enhanced_feature_importance.png')
plt.close()

# Save selected features and their importance scores
composite_features = [
    'Vol_Body_Interaction', 'Price_Action_Signal', 'Market_Impact',
    'SR_Strength', 'Volatility_Regime', 'Enhanced_Volume_Impact',
    'Trend_Strength', 'SR_Break_Potential'
]

feature_importance_df = pd.DataFrame({
    'Feature': original_importance.index,
    'Importance': original_importance.values,
    'Is_Composite': original_importance.index.isin(composite_features)
})
feature_importance_df.to_csv('enhanced_feature_selection.csv', index=False)

# Print feature importance summary
print("\nTop 10 Most Important Features:")
for i, (feature, importance) in enumerate(list(original_importance.items())[:10], 1):
    feature_type = "Composite" if feature in composite_features else "Original"
    print(f"{i}. {feature} ({feature_type}): {importance:.4f}")

# Calculate and plot feature correlations for top features
top_features = original_importance.index[:10]
correlation_matrix = df[top_features].corr()

plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.title('Correlation Heatmap of Enhanced Features')
plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
plt.yticks(range(len(top_features)), top_features)
plt.tight_layout()
plt.savefig('enhanced_feature_correlation.png')
plt.close()

print("\nEnhanced feature analysis complete. Check 'enhanced_feature_importance.png' and 'enhanced_feature_correlation.png' for visualizations.")
