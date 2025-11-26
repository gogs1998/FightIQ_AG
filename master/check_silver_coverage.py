import pandas as pd

def check_silver_coverage():
    print("Loading Silver data...")
    df = pd.read_csv('d:/AntiGravity/FightIQ/UFC_full_data_silver.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['Year'] = df['event_date'].dt.year
    
    print(f"Total rows: {len(df)}")
    
    # Check coverage by year
    print("\n--- Silver Odds Coverage by Year ---")
    # Check if odds columns exist and are not null
    # Silver might use different column names? Let's assume f_1_odds/f_2_odds based on previous interactions
    # But let's verify columns first just in case
    odds_cols = ['R_odds', 'B_odds'] if 'R_odds' in df.columns else ['f_1_odds', 'f_2_odds']
    
    if odds_cols[0] not in df.columns:
        print(f"Could not find standard odds columns. Available columns: {df.columns.tolist()}")
        return

    print(f"Using odds columns: {odds_cols}")
    
    missing_odds = df[df[odds_cols[0]].isna() | df[odds_cols[1]].isna()]
    missing_by_year = missing_odds.groupby('Year').size()
    total_by_year = df.groupby('Year').size()
    
    summary = pd.DataFrame({'Total': total_by_year, 'Missing': missing_by_year}).fillna(0)
    summary['Missing %'] = (summary['Missing'] / summary['Total'] * 100).round(1)
    
    # Print pre-2015 specifically as that's where Master was missing data
    print(summary.head(25))

if __name__ == "__main__":
    check_silver_coverage()
