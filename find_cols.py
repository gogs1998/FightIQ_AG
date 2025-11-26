import pandas as pd

df = pd.read_csv('UFC_full_data_silver.csv', nrows=5)
cols = list(df.columns)
fighter_cols = [c for c in cols if 'fighter' in c.lower() or 'name' in c.lower()]
print("Columns related to fighter/name:")
print(fighter_cols)
