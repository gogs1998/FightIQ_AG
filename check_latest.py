import pandas as pd

df = pd.read_csv('UFC_full_data_golden.csv')
df['event_date'] = pd.to_datetime(df['event_date'])
print(f"Latest date in dataset: {df['event_date'].max()}")

# Check if there are any rows with no winner (upcoming?)
upcoming = df[df['winner'].isnull()]
print(f"Rows with null winner: {len(upcoming)}")
if len(upcoming) > 0:
    print(upcoming[['event_date', 'f_1_name', 'f_2_name']].head())
