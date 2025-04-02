from GatherData import fetch_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

output_dataset = "aapl_df_indicators_preprocessed.csv"

# Load the data
# aapl_df = fetch_data("AAPL", interval="1min", outputsize="full")
aapl_df = pd.read_csv("aapl_df_indicators.csv", index_col=0)
# aapl_df.to_csv("aapl_df_indicators.csv")
# quit()

# Step 1: Data Preprocessing
print("Starting data preprocessing...")

# Identify column types
datetime_columns = aapl_df.select_dtypes(include=['datetime64']).columns
numerical_columns = aapl_df.select_dtypes(include=['float64', 'int64']).columns
boolean_columns = aapl_df.select_dtypes(include=['bool']).columns
categorical_columns = aapl_df.select_dtypes(include=['object']).columns

print("\nColumn types found:")
print(f"Datetime columns: {len(datetime_columns)}")
print(f"Numerical columns: {len(numerical_columns)}")
print(f"Boolean columns: {len(boolean_columns)}")
print(f"Categorical columns: {len(categorical_columns)}")

# Handle missing values using appropriate methods for each type
print("\nHandling missing values...")

# For numerical columns: use median imputation
if len(numerical_columns) > 0:
    aapl_df[numerical_columns] = aapl_df[numerical_columns].fillna(aapl_df[numerical_columns].median())

# For categorical columns: use mode imputation
if len(categorical_columns) > 0:
    aapl_df[categorical_columns] = aapl_df[categorical_columns].fillna(aapl_df[categorical_columns].mode().iloc[0])

# For datetime columns: use forward fill then backward fill
if len(datetime_columns) > 0:
    aapl_df[datetime_columns] = aapl_df[datetime_columns].fillna(method='ffill').fillna(method='bfill')

# Convert boolean features to numerical (1/0)
print("Converting boolean features to numerical...")
if len(boolean_columns) > 0:
    aapl_df[boolean_columns] = aapl_df[boolean_columns].replace({True: 1, False: 0})

# Handle infinite values in numerical columns
print("Handling infinite values...")
if len(numerical_columns) > 0:
    # Replace infinite values with NaN
    aapl_df[numerical_columns] = aapl_df[numerical_columns].replace([np.inf, -np.inf], np.nan)
    # Fill NaN values with median
    aapl_df[numerical_columns] = aapl_df[numerical_columns].fillna(aapl_df[numerical_columns].median())

# Standardize numerical features
print("Standardizing numerical features...")
if len(numerical_columns) > 0:
    scaler = StandardScaler()
    # Ensure all values are finite before standardization
    aapl_df[numerical_columns] = scaler.fit_transform(aapl_df[numerical_columns])

# Save the preprocessed data
aapl_df.to_csv(output_dataset, index=False)
print(f"\nPreprocessed data has been saved to {output_dataset}")

# Print summary statistics
print("\nPreprocessing Summary:")
print(f"Total number of records: {len(aapl_df)}")
print(f"Number of features: {len(aapl_df.columns)}")
print("\nFeature types:")
print(aapl_df.dtypes.value_counts())
print("\nFirst few rows of preprocessed data:")
print(aapl_df.head())

