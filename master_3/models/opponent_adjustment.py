import pandas as pd
import numpy as np

def apply_opponent_adjustment(df, stat_cols, elo_col='dynamic_elo'):
    """
    Adjusts statistics based on the strength of opponents faced.
    Adj_Stat = Raw_Stat * (Opponent_Elo / Avg_Elo)
    """
    df = df.copy()
    
    # 1. Calculate Global Average Elo
    avg_elo = df[f'{elo_col}_f1'].mean()
    
    for col in stat_cols:
        # We need to know which fighter this stat belongs to.
        # Assuming format 'f_1_stat' and 'f_2_stat'
        
        base_stat = col.replace('f_1_', '').replace('f_2_', '')
        
        if col.startswith('f_1_') or col.endswith('_f_1'):
            # F1's stat. Adjust by F2's Elo (the opponent).
            opp_elo = df[f'{elo_col}_f2']
            df[f'{col}_adj'] = df[col] * (opp_elo / avg_elo)
            
        elif col.startswith('f_2_') or col.endswith('_f_2'):
            # F2's stat. Adjust by F1's Elo.
            opp_elo = df[f'{elo_col}_f1']
            df[f'{col}_adj'] = df[col] * (opp_elo / avg_elo)
            
    return df
