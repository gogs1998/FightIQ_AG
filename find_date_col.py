import pandas as pd

df = pd.read_csv('UFC_full_data_golden.csv', nrows=5)
cols = list(df.columns)
date_cols = [c for c in cols if 'date' in c.lower() or 'time' in c.lower() or 'year' in c.lower()]
print(f"Date/Time related columns: {date_cols}")
