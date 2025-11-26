import pandas as pd
import os

f = 'UFC_full_data_golden.csv'
path = os.path.join(os.getcwd(), f)

print(f"--- Inspecting {f} ---")
try:
    df_head = pd.read_csv(path, nrows=5)
    print(f"Number of Columns: {len(df_head.columns)}")
    print(f"Shape: {df_head.shape}")
    print(f"First 5 Columns: {list(df_head.columns)[:5]}")
except Exception as e:
    print(f"Error: {e}")
