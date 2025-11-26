import pandas as pd
try:
    df = pd.read_csv('d:/AntiGravity/FightIQ/master/data/training_data.csv')
    print(f"Length: {len(df)}")
except Exception as e:
    print(f"Error: {e}")
