import pandas as pd

def check_test_odds():
    print("Loading data...")
    df = pd.read_csv('UFC_full_data_golden.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Create Target (same filtering as training)
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    
    # Split (Test set only)
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Test Set Period: {test_df['event_date'].min().date()} to {test_df['event_date'].max().date()}")
    print(f"Total Fights in Test Set: {len(test_df)}")
    
    # Check missing odds
    missing_f1 = test_df['f_1_odds'].isnull().sum()
    missing_f2 = test_df['f_2_odds'].isnull().sum()
    
    # Usually if one is missing, both are missing, but let's check rows where ANY is missing
    missing_any = test_df[test_df['f_1_odds'].isnull() | test_df['f_2_odds'].isnull()]
    
    print(f"\nMissing Odds Statistics:")
    print(f"Fights with missing odds: {len(missing_any)}")
    print(f"Percentage missing: {len(missing_any) / len(test_df):.2%}")
    
    if len(missing_any) > 0:
        print("\nSample of fights with missing odds:")
        print(missing_any[['event_date', 'f_1_name', 'f_2_name']].head())

if __name__ == "__main__":
    check_test_odds()
