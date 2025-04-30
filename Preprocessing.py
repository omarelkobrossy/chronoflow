from GatherData import fetch_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

output_dataset = "DB/TSLA_15min_indicators_preprocessed.csv"

# Load the data
# df = fetch_data("AAPL", interval="1min", outputsize="full")
df = pd.read_csv("DB/TSLA_15min_indicators.csv", index_col=0)
# df.to_csv("df_indicators.csv")
# quit()

# Step 1: Data Preprocessing
print("Starting data preprocessing...")

# Store original price for backtesting
df['Close_Original'] = df['Close'].copy()

# Identify column types
datetime_columns = df.select_dtypes(include=['datetime64']).columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
boolean_columns = df.select_dtypes(include=['bool']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Remove Close_Original from numerical columns to avoid scaling it
numerical_columns = numerical_columns.drop('Close_Original')

print("\nColumn types found:")
print(f"Datetime columns: {len(datetime_columns)}")
print(f"Numerical columns: {len(numerical_columns)}")
print(f"Boolean columns: {len(boolean_columns)}")
print(f"Categorical columns: {len(categorical_columns)}")

# Handle missing values using appropriate methods for each type
print("\nHandling missing values...")

# For numerical columns: use median imputation
if len(numerical_columns) > 0:
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# For categorical columns: use mode imputation
if len(categorical_columns) > 0:
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# For datetime columns: use forward fill then backward fill
if len(datetime_columns) > 0:
    df[datetime_columns] = df[datetime_columns].fillna(method='ffill').fillna(method='bfill')

# Convert boolean features to numerical (1/0)
print("Converting boolean features to numerical...")
if len(boolean_columns) > 0:
    df[boolean_columns] = df[boolean_columns].replace({True: 1, False: 0})

# Handle infinite values in numerical columns
print("Handling infinite values...")
if len(numerical_columns) > 0:
    # Replace infinite values with NaN
    df[numerical_columns] = df[numerical_columns].replace([np.inf, -np.inf], np.nan)
    # Fill NaN values with median
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Standardize numerical features using rolling window to avoid forward bias
print("Standardizing numerical features...")
if len(numerical_columns) > 0:
    # Create a copy of the dataframe for standardization
    standardized_df = df[numerical_columns].copy()
    
    # Apply standardization with expanding window for first 100 records
    window_size = 100  # Adjust window size as needed
    for col in numerical_columns:
        # Calculate expanding mean and std for first window_size records
        expanding_mean = standardized_df[col].expanding(min_periods=1).mean()
        expanding_std = standardized_df[col].expanding(min_periods=1).std()
        
        # Calculate rolling mean and std for remaining records using only past data
        rolling_mean = standardized_df[col].shift(1).rolling(window=window_size, min_periods=1).mean()
        rolling_std = standardized_df[col].shift(1).rolling(window=window_size, min_periods=1).std()
        
        # Combine expanding and rolling statistics
        mean_combined = expanding_mean.copy()
        mean_combined[window_size:] = rolling_mean[window_size:]
        
        std_combined = expanding_std.copy()
        std_combined[window_size:] = rolling_std[window_size:]
        
        # Standardize using combined statistics
        standardized_df[col] = (standardized_df[col] - mean_combined) / (std_combined + 1e-6)
    
    # Replace original columns with standardized values
    df[numerical_columns] = standardized_df

    # Print information about NaN values
    nan_counts = standardized_df.isna().sum()
    print("\nNaN counts after standardization:")
    print(nan_counts[nan_counts > 0])
    
    # Verify no NaN values remain
    if standardized_df.isna().any().any():
        print("\nWarning: Some NaN values remain after standardization")
        # Fill any remaining NaN values with 0 (they should be very few if any)
        standardized_df = standardized_df.fillna(0)
        df[numerical_columns] = standardized_df

    # Drop the first row after it's been used for calculations
    df = df.iloc[1:].copy()
    print("\nDropped first row after standardization calculations")

# Save the preprocessed data
df.to_csv(output_dataset)
print(f"\nPreprocessed data has been saved to {output_dataset}")

# Print summary statistics
print("\nPreprocessing Summary:")
print(f"Total number of records: {len(df)}")
print(f"Number of features: {len(df.columns)}")
print("\nFeature types:")
print(df.dtypes.value_counts())
print("\nFirst few rows of preprocessed data:")
print(df.head())

