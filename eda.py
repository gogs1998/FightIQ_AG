import pandas as pd
import os

# Load Silver data
file_path = 'UFC_full_data_silver.csv'
print(f"Loading {file_path}...")
df = pd.read_csv(file_path)

print(f"Dataset Shape: {df.shape}")

# Check Target Variable
if 'winner' in df.columns:
    print("\nTarget Variable ('winner') Distribution:")
    print(df['winner'].value_counts(normalize=True))
else:
    print("\n'winner' column not found!")

# Check for missing values
print("\nMissing Values Summary (Top 10 columns with most missing):")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing.head(10))

# Check data types
print("\nData Types:")
print(df.dtypes.value_counts())

# Preview a few interesting columns if they exist
cols_of_interest = ['fighter_1', 'fighter_2', 'weight_class', 'date']
existing_cols = [c for c in cols_of_interest if c in df.columns]
if existing_cols:
    print(f"\nPreview of {existing_cols}:")
    print(df[existing_cols].head())
