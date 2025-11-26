import pandas as pd

def check_missing():
    print("Checking for missing odds in 2025...")
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Filter 2025
    mask_2025 = df['event_date'] >= '2025-01-01'
    df_2025 = df[mask_2025].copy()
    
    print(f"Total fights in 2025: {len(df_2025)}")
    
    # Check for missing odds (0 or NaN)
    missing_mask = (df_2025['f_1_odds'].isna()) | (df_2025['f_1_odds'] == 0) | \
                   (df_2025['f_2_odds'].isna()) | (df_2025['f_2_odds'] == 0)
                   
    missing_df = df_2025[missing_mask]
    
    print(f"Fights with missing odds: {len(missing_df)}")
    
    if len(missing_df) > 0:
        events = missing_df[['event_date', 'event_name', 'target']].drop_duplicates().sort_values('event_date')
        print(f"\nUnique Events with missing odds: {len(events)}")
        print("\nEvents list (First 20):")
        for _, row in events.head(20).iterrows():
            print(f"{row['event_date'].date()} - {row['event_name']} (Target: {row['target']})")
            
if __name__ == "__main__":
    check_missing()
