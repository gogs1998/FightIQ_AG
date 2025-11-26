import pandas as pd

def check_golden_coverage():
    print("Loading Golden data...")
    # Note: Golden file is likely larger, so we'll read carefully
    try:
        df = pd.read_csv('d:/AntiGravity/FightIQ/UFC_full_data_golden.csv')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Check for date column
    date_col = 'date' if 'date' in df.columns else 'event_date'
    if date_col not in df.columns:
        print(f"Date column not found. Available: {df.columns.tolist()[:10]}...")
        return
        
    df[date_col] = pd.to_datetime(df[date_col])
    df['Year'] = df[date_col].dt.year
    
    print(f"Total rows: {len(df)}")
    
    # Check coverage by year
    print("\n--- Golden Odds Coverage by Year ---")
    
    # Identify odds columns
    odds_cols = []
    candidates = [['R_odds', 'B_odds'], ['f_1_odds', 'f_2_odds'], ['odds_1', 'odds_2']]
    
    for pair in candidates:
        if pair[0] in df.columns and pair[1] in df.columns:
            odds_cols = pair
            break
            
    if not odds_cols:
        print(f"Could not find standard odds columns. Available columns: {df.columns.tolist()[:20]}")
        return

    print(f"Using odds columns: {odds_cols}")
    
    missing_odds = df[df[odds_cols[0]].isna() | df[odds_cols[1]].isna()]
    missing_by_year = missing_odds.groupby('Year').size()
    total_by_year = df.groupby('Year').size()
    
    summary = pd.DataFrame({'Total': total_by_year, 'Missing': missing_by_year}).fillna(0)
    summary['Missing %'] = (summary['Missing'] / summary['Total'] * 100).round(1)
    
    # Print pre-2015 specifically
    print(summary.head(25))

if __name__ == "__main__":
    check_golden_coverage()
