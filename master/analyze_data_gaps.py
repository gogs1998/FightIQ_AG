import pandas as pd
import numpy as np

def analyze_gaps_and_swaps():
    print("Loading data...")
    df = pd.read_csv('d:/AntiGravity/FightIQ/master/data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['Year'] = df['event_date'].dt.year
    
    # 1. Missing Odds Analysis
    print("\n--- Missing Odds by Year ---")
    missing_odds = df[df['f_1_odds'].isna() | df['f_2_odds'].isna()]
    missing_by_year = missing_odds.groupby('Year').size()
    total_by_year = df.groupby('Year').size()
    
    summary = pd.DataFrame({'Total': total_by_year, 'Missing': missing_by_year}).fillna(0)
    summary['Missing %'] = (summary['Missing'] / summary['Total'] * 100).round(1)
    print(summary)
    
    # 2. Swapped Odds Investigation
    print("\n--- Swapped Odds Investigation ---")
    # Load Silver for comparison (needed to reproduce the 'swap' logic)
    df_silver = pd.read_csv('d:/AntiGravity/FightIQ/UFC_full_data_silver.csv')
    df_silver['event_date'] = pd.to_datetime(df_silver['event_date'])
    
    def clean_name(name):
        if pd.isna(name): return ""
        return name.strip().lower().replace(" ", "")

    df['join_key'] = df['event_date'].dt.strftime('%Y-%m-%d') + "_" + df['f_1_name'].apply(clean_name)
    df_silver['join_key'] = df_silver['event_date'].dt.strftime('%Y-%m-%d') + "_" + df_silver['f_1_name'].apply(clean_name)
    
    merged = pd.merge(df, df_silver, on='join_key', how='inner', suffixes=('_master', '_silver'))
    
    # Calculate diffs
    merged['diff_normal'] = np.abs(merged['f_1_odds_master'] - merged['f_1_odds_silver'])
    merged['diff_swap'] = np.abs(merged['f_1_odds_master'] - merged['f_2_odds_silver'])
    
    # Define "Real Swap": Normal diff is high (> 0.2), Swap diff is low (< 0.1)
    real_swaps = merged[(merged['diff_normal'] > 0.2) & (merged['diff_swap'] < 0.1)]
    
    # Define "Close Fights": Both diffs are low (e.g. odds are 1.9 vs 1.9)
    close_fights = merged[(merged['diff_normal'] < 0.2) & (merged['diff_swap'] < 0.2)]
    
    print(f"Total Matches Checked: {len(merged)}")
    print(f"Potential Swaps (Raw Metric): {len(merged[merged['diff_swap'] < 0.1])}")
    print(f"True Swaps (High Normal Diff, Low Swap Diff): {len(real_swaps)}")
    print(f"Close Fights (Low Normal Diff, Low Swap Diff): {len(close_fights)}")
    
    if len(real_swaps) > 0:
        print("\nSample True Swaps:")
        cols = ['f_1_name_master', 'f_2_name_master', 'f_1_odds_master', 'f_2_odds_master', 'f_1_odds_silver', 'f_2_odds_silver']
        print(real_swaps[cols].head(10).to_string())

if __name__ == "__main__":
    analyze_gaps_and_swaps()
