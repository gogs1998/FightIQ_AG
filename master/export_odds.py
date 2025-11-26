import pandas as pd

def export_odds():
    source_path = 'd:/AntiGravity/FightIQ/master/data/training_data.csv'
    output_path = 'd:/AntiGravity/FightIQ/UFC_betting_odds.csv'
    
    print(f"Reading from {source_path}...")
    df = pd.read_csv(source_path)
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Filter for 2020-2025
    mask = (df['event_date'] >= '2020-01-01') & (df['event_date'] <= '2025-12-31')
    df_filtered = df[mask].copy()
    
    print(f"Filtered {len(df_filtered)} rows from 2020-2025.")
    
    # Select relevant columns
    cols = ['event_name', 'event_date', 'f_1_name', 'f_2_name', 'f_1_odds', 'f_2_odds']
    # Add result columns if available for context, but user asked for odds
    if 'winner' in df.columns:
        cols.append('winner')
    
    df_export = df_filtered[cols]
    
    print(f"Saving to {output_path}...")
    df_export.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    export_odds()
