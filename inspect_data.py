import pandas as pd
import os

files = ['UFC_full_data_silver.csv', 'UFC_full_data_golden.csv']

for f in files:
    path = os.path.join(os.getcwd(), f)
    if os.path.exists(path):
        print(f"--- Inspecting {f} ---")
        try:
            df_head = pd.read_csv(path, nrows=5)
            print(f"Number of Columns: {len(df_head.columns)}")
            print(f"First 10 Columns: {list(df_head.columns)[:10]}")
            print(f"Shape (first 5 rows): {df_head.shape}")
            
            # Count lines for total rows (approximate)
            with open(path, 'r', encoding='utf-8') as file:
                line_count = sum(1 for _ in file)
            print(f"Total Rows: {line_count}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
        print("\n")
