import pandas as pd
import numpy as np

def clean_name(name):
    if pd.isna(name): return ""
    return name.strip().lower().replace(" ", "")

def verify_alignment():
    print("Loading datasets...")
    # Load Master Data (Updated)
    df_master = pd.read_csv('d:/AntiGravity/FightIQ/master/data/training_data.csv')
    df_master['event_date'] = pd.to_datetime(df_master['event_date'])
    
    # Load Silver Data (Reference)
    df_silver = pd.read_csv('d:/AntiGravity/FightIQ/UFC_full_data_silver.csv')
    df_silver['event_date'] = pd.to_datetime(df_silver['event_date'])
    
    print(f"Master Rows: {len(df_master)}")
    print(f"Silver Rows: {len(df_silver)}")
    
    # Create join keys
    df_master['join_key'] = df_master['event_date'].dt.strftime('%Y-%m-%d') + "_" + df_master['f_1_name'].apply(clean_name)
    df_silver['join_key'] = df_silver['event_date'].dt.strftime('%Y-%m-%d') + "_" + df_silver['f_1_name'].apply(clean_name)
    
    # Merge
    print("Merging datasets...")
    merged = pd.merge(df_master, df_silver, on='join_key', how='inner', suffixes=('_master', '_silver'))
    # Compare Odds
    merged['diff_1'] = np.abs(merged['f_1_odds_master'] - merged['f_1_odds_silver'])
    merged['diff_2'] = np.abs(merged['f_2_odds_master'] - merged['f_2_odds_silver'])
    
    # Filter for rows where both have odds
    valid_comparison = merged.dropna(subset=['f_1_odds_master', 'f_1_odds_silver', 'f_2_odds_master', 'f_2_odds_silver'])
    
    # Metrics
    mae_1 = valid_comparison['diff_1'].mean()
    mae_2 = valid_comparison['diff_2'].mean()
    corr_1 = valid_comparison['f_1_odds_master'].corr(valid_comparison['f_1_odds_silver'])
    
    # Check for swapped odds (F1 Master ~ F2 Silver)
    merged['diff_swap'] = np.abs(merged['f_1_odds_master'] - merged['f_2_odds_silver'])
    swap_candidates = merged[merged['diff_swap'] < 0.1]

    # Log correlation (more robust to outliers in odds)
    valid_comparison['log_odds_1_master'] = np.log(valid_comparison['f_1_odds_master'])
    valid_comparison['log_odds_1_silver'] = np.log(valid_comparison['f_1_odds_silver'])
    log_corr = valid_comparison['log_odds_1_master'].corr(valid_comparison['log_odds_1_silver'])
    
    # Check for major discrepancies (> 0.5 difference in decimal odds)
    major_diffs = valid_comparison[(valid_comparison['diff_1'] > 0.5) | (valid_comparison['diff_2'] > 0.5)]

    with open('verification_results.txt', 'w') as f:
        f.write(f"Master Rows: {len(df_master)}\n")
        f.write(f"Silver Rows: {len(df_silver)}\n")
        f.write(f"Matched Rows: {len(merged)}\n")
        f.write(f"Rows with valid odds in both: {len(valid_comparison)}\n\n")
        
        f.write(f"Alignment Metrics:\n")
        f.write(f"Mean Absolute Error (Fighter 1): {mae_1:.4f}\n")
        f.write(f"Mean Absolute Error (Fighter 2): {mae_2:.4f}\n")
        f.write(f"Correlation (Fighter 1): {corr_1:.4f}\n")
        f.write(f"Log Odds Correlation: {log_corr:.4f}\n\n")
        
        f.write(f"Potential Swapped Odds Rows: {len(swap_candidates)}\n")
        
        # Check for Format Mismatch (American vs Decimal)
        # American odds are usually > 100 or < -100. Decimal odds are usually < 10.
        format_mismatch = valid_comparison[((valid_comparison['f_1_odds_master'] > 50) & (valid_comparison['f_1_odds_silver'] < 20)) |
                                           ((valid_comparison['f_1_odds_master'] < 20) & (valid_comparison['f_1_odds_silver'] > 50))]
        f.write(f"Potential Format Mismatch Rows: {len(format_mismatch)}\n\n")
        
        f.write("Sample Pairs (Master vs Silver):\n")
        f.write(valid_comparison[['f_1_odds_master', 'f_1_odds_silver']].head(20).to_string() + "\n\n")
        
        f.write(f"Major Discrepancies (> 0.5): {len(major_diffs)}\n")
        if len(major_diffs) > 0:
            f.write("Top 20 Discrepancies:\n")
            f.write(major_diffs[['event_date_master', 'f_1_name_master', 'f_1_odds_master', 'f_1_odds_silver', 'f_2_odds_master', 'f_2_odds_silver']].head(20).to_string() + "\n")

    print("Verification complete. Results saved to verification_results.txt")

if __name__ == "__main__":
    verify_alignment()
