import pandas as pd
import numpy as np
import json
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from features.pear import fit_pear
from features.dynamic_elo import calculate_dynamic_elo
from features.common_opponents import calculate_common_opponent_features
from features.stoppage import calculate_stoppage_features
from features.chin import calculate_chin_features

def generate_all_features():
    print("=== Generating Features for Master 3 Pipeline ===")
    
    # 1. Load Data
    print("Loading data...")
    if os.path.exists('data/training_data.csv'):
        df = pd.read_csv('data/training_data.csv')
    else:
        # Fallback to root data if not in local data dir
        df = pd.read_csv('../training_data.csv')
        
    print(f"Loaded {len(df)} fights.")
    
    # 2. Dynamic Elo
    print("Calculating Dynamic Elo...")
    df = calculate_dynamic_elo(df)
    
    # 3. Common Opponents
    print("Calculating Common Opponents...")
    df = calculate_common_opponent_features(df)
    
    # 4. Stoppage Propensity
    print("Calculating Stoppage Propensity...")
    df = calculate_stoppage_features(df)
    
    # 5. Chin Health
    print("Calculating Chin Health...")
    df = calculate_chin_features(df)
    
    # 6. PEAR (Requires Round Data)
    # We need to load round data. Assuming it's in 'data/round_data.csv' or we extract it.
    # For now, if round data is missing, we skip PEAR or try to extract from golden if available.
    print("Calculating PEAR features...")
    try:
        # Try to load golden data to extract rounds
        # This is a bit hacky, but we need round-level data for PEAR
        if os.path.exists('../UFC_full_data_golden.csv'):
            golden = pd.read_csv('../UFC_full_data_golden.csv')
            # from experimental.features.round_dynamics import prep_rounds # Reuse prep logic if possible
            # Or just implement simple prep here
            
            # Simple prep
            long_rows = []
            for idx, row in golden.iterrows():
                fid = f"{row['event_date']}_{row['f_1_name']}_{row['f_2_name']}"
                for r in range(1, 6):
                    s_land = row.get(f'f_1_r{r}_sig_strikes_succ')
                    if pd.notna(s_land):
                        long_rows.append({
                            'fighter_id': row['f_1_name'],
                            'fight_id': fid,
                            'round': r,
                            'sig_str_diff': row.get(f'f_1_r{r}_sig_strikes_succ', 0) - row.get(f'f_2_r{r}_sig_strikes_succ', 0),
                            'opp_sig_str_per_min': row.get(f'f_2_r{r}_sig_strikes_succ', 0) # Proxy for pace
                        })
                        long_rows.append({
                            'fighter_id': row['f_2_name'],
                            'fight_id': fid,
                            'round': r,
                            'sig_str_diff': row.get(f'f_2_r{r}_sig_strikes_succ', 0) - row.get(f'f_1_r{r}_sig_strikes_succ', 0),
                            'opp_sig_str_per_min': row.get(f'f_1_r{r}_sig_strikes_succ', 0)
                        })
            
            df_rounds = pd.DataFrame(long_rows)
            pear_stats = fit_pear(df_rounds)
            
            if not pear_stats.empty:
                # Merge PEAR stats back to main df
                # PEAR gives one row per fighter (static trait).
                # We merge on f_1_name and f_2_name
                
                pear_stats.columns = ['fighter_id', 'pear_beta_pace', 'pear_beta_lag', 'pear_n_rounds']
                
                # Merge F1
                df = df.merge(pear_stats, left_on='f_1_name', right_on='fighter_id', how='left')
                df = df.rename(columns={
                    'pear_beta_pace': 'f_1_pear_pace',
                    'pear_beta_lag': 'f_1_pear_lag',
                    'pear_n_rounds': 'f_1_pear_n'
                }).drop(columns=['fighter_id'])
                
                # Merge F2
                df = df.merge(pear_stats, left_on='f_2_name', right_on='fighter_id', how='left')
                df = df.rename(columns={
                    'pear_beta_pace': 'f_2_pear_pace',
                    'pear_beta_lag': 'f_2_pear_lag',
                    'pear_n_rounds': 'f_2_pear_n'
                }).drop(columns=['fighter_id'])
                
                # Diffs
                df['diff_pear_pace'] = df['f_1_pear_pace'] - df['f_2_pear_pace']
                df['diff_pear_lag'] = df['f_1_pear_lag'] - df['f_2_pear_lag']
                
                print("PEAR features added.")
            else:
                print("PEAR stats empty.")
        else:
            print("Golden data not found, skipping PEAR.")
    except Exception as e:
        print(f"Error calculating PEAR: {e}")
        
    # 7. Save Enhanced Data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/training_data_enhanced.csv', index=False)
    print("Saved to data/training_data_enhanced.csv")
    
    # 8. Update features.json
    # Load original features
    with open('features.json', 'r') as f:
        original_features = json.load(f)
        
    new_features = [
        # Dynamic Elo
        'dynamic_elo_f1', 'dynamic_elo_f2', 'diff_dynamic_elo',
        # Common Opponents
        'triangle_score', 'n_common_opponents', 'common_win_pct_diff',
        # Stoppage
        'f_1_finish_rate', 'f_1_been_finished_rate', 'f_1_avg_time',
        'f_2_finish_rate', 'f_2_been_finished_rate', 'f_2_avg_time',
        'diff_finish_rate', 'diff_been_finished_rate', 'diff_avg_time',
        # Chin
        'f_1_chin_score', 'f_2_chin_score', 'diff_chin_score',
        # PEAR
        'f_1_pear_pace', 'f_1_pear_lag',
        'f_2_pear_pace', 'f_2_pear_lag',
        'diff_pear_pace', 'diff_pear_lag'
    ]
    
    # Filter only those present in df
    final_new = [c for c in new_features if c in df.columns]
    
    all_features = list(set(original_features + final_new))
    
    with open('features_enhanced.json', 'w') as f:
        json.dump(all_features, f, indent=4)
        
    print(f"Updated features list. Total features: {len(all_features)}")

if __name__ == "__main__":
    generate_all_features()
