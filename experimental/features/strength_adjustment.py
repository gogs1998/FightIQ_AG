import pandas as pd
import numpy as np

def calculate_adjusted_stats(df, elo_col_f1='elo_f1', elo_col_f2='elo_f2'):
    """
    Calculate strength-adjusted stats for each fighter.
    Adjusted Stat = Raw Stat * (Opponent Elo / 1500)
    Then compute expanding mean.
    """
    df = df.copy()
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
    # Define stats to adjust
    # We need the per-fight stats, not the cumulative averages.
    # Assuming columns like 'f_1_sig_strikes_succ' exist.
    
    stats_map = {
        'sig_strikes_succ': 'adj_slpm', # We'll convert to per minute later or just use count
        'total_strikes_succ': 'adj_ts',
        'takedown_succ': 'adj_td',
        'knockdowns': 'adj_kd'
    }
    
    # We need to handle the fact that we might not have 'minutes' easily available for every row 
    # to normalize to "per minute" right away.
    # But we can just adjust the raw count, and then the average of adjusted counts 
    # is a proxy for "Adjusted Volume".
    
    # 1. Compute Adjusted Stats for each row (fight)
    # 1. Compute Adjusted Stats for each row (fight)
    # Normalize by duration if available
    if 'fight_duration_minutes' in df.columns:
        # Avoid division by zero
        duration = df['fight_duration_minutes'].replace(0, 1)
        df['slpm_fight'] = df['f_1_sig_strikes_succ'] / duration
        df['ts_fight'] = df['f_1_total_strikes_succ'] / duration
        df['td_fight'] = df['f_1_takedown_succ'] / duration
        df['kd_fight'] = df['f_1_knockdowns'] / duration
        
        # F2
        df['f2_slpm_fight'] = df['f_2_sig_strikes_succ'] / duration
        df['f2_ts_fight'] = df['f_2_total_strikes_succ'] / duration
        df['f2_td_fight'] = df['f_2_takedown_succ'] / duration
        df['f2_kd_fight'] = df['f_2_knockdowns'] / duration
    else:
        # Fallback to raw counts
        df['slpm_fight'] = df['f_1_sig_strikes_succ']
        df['ts_fight'] = df['f_1_total_strikes_succ']
        df['td_fight'] = df['f_1_takedown_succ']
        df['kd_fight'] = df['f_1_knockdowns']
        
        df['f2_slpm_fight'] = df['f_2_sig_strikes_succ']
        df['f2_ts_fight'] = df['f_2_total_strikes_succ']
        df['f2_td_fight'] = df['f_2_takedown_succ']
        df['f2_kd_fight'] = df['f_2_knockdowns']

    for raw_base, new_base in stats_map.items():
        # F1 stats
        if raw_base == 'sig_strikes_succ':
            val = df['slpm_fight']
        elif raw_base == 'total_strikes_succ':
            val = df['ts_fight']
        elif raw_base == 'takedown_succ':
            val = df['td_fight']
        elif raw_base == 'knockdowns':
            val = df['kd_fight']
        else:
            val = df.get(f"f_1_{raw_base}", 0)
            
        # F1's performance is weighted by F2's Elo
        opp_elo = df[elo_col_f2].fillna(1500)
        df[f"f_1_{new_base}_raw"] = val * (opp_elo / 1500.0)
            
        # F2 stats
        if raw_base == 'sig_strikes_succ':
            val2 = df['f2_slpm_fight']
        elif raw_base == 'total_strikes_succ':
            val2 = df['f2_ts_fight']
        elif raw_base == 'takedown_succ':
            val2 = df['f2_td_fight']
        elif raw_base == 'knockdowns':
            val2 = df['f2_kd_fight']
        else:
            val2 = df.get(f"f_2_{raw_base}", 0)

        # F2's performance is weighted by F1's Elo
        opp_elo = df[elo_col_f1].fillna(1500)
        df[f"f_2_{new_base}_raw"] = val2 * (opp_elo / 1500.0)
            
    # 2. Compute Expanding Mean per Fighter
    # We need to reconstruct the history.
    
    # Stack F1 and F2
    cols_to_agg = [f"{new_base}_raw" for new_base in stats_map.values()]
    
    f1_cols = ['f_1_name', 'event_date'] + [f"f_1_{c}" for c in cols_to_agg]
    f2_cols = ['f_2_name', 'event_date'] + [f"f_2_{c}" for c in cols_to_agg]
    
    # Rename to common
    rename_f1 = {f"f_1_{c}": c for c in cols_to_agg}
    rename_f1['f_1_name'] = 'fighter_id'
    
    rename_f2 = {f"f_2_{c}": c for c in cols_to_agg}
    rename_f2['f_2_name'] = 'fighter_id'
    
    df_f1 = df[f1_cols].rename(columns=rename_f1)
    df_f2 = df[f2_cols].rename(columns=rename_f2)
    
    history = pd.concat([df_f1, df_f2]).sort_values('event_date')
    
    # Group by fighter and expanding mean
    # Shift 1 to avoid leakage (stats from current fight shouldn't be used for current fight prediction)
    
    features_per_fighter = {}
    
    print("Calculating expanding adjusted stats...")
    for fid, g in history.groupby('fighter_id'):
        g = g.sort_values('event_date')
        # Expanding mean of raw adjusted stats
        # We only care about the stats columns
        stats = g[cols_to_agg].expanding().mean().shift(1)
        
        # Merge back date
        stats['event_date'] = g['event_date']
        features_per_fighter[fid] = stats
        
    # 3. Merge back to main DF
    # Similar to round dynamics, use merge_asof or just lookup
    
    # Prepare lookup table
    lookup_dfs = []
    for fid, stats in features_per_fighter.items():
        stats['fighter_id'] = fid
        lookup_dfs.append(stats)
        
    if not lookup_dfs:
        return df
        
    df_lookup = pd.concat(lookup_dfs).sort_values('event_date').dropna(subset=cols_to_agg)
    
    # Merge F1
    df = pd.merge_asof(
        df.sort_values('event_date'),
        df_lookup.sort_values('event_date'),
        on='event_date',
        left_by='f_1_name',
        right_by='fighter_id',
        direction='backward'
    )
    
    # Rename
    rename_back_f1 = {c: f"f_1_{c}_avg" for c in cols_to_agg}
    df = df.rename(columns=rename_back_f1)
    df = df.drop(columns=['fighter_id'], errors='ignore')
    
    # Merge F2
    df = pd.merge_asof(
        df.sort_values('event_date'),
        df_lookup.sort_values('event_date'),
        on='event_date',
        left_by='f_2_name',
        right_by='fighter_id',
        direction='backward'
    )
    
    rename_back_f2 = {c: f"f_2_{c}_avg" for c in cols_to_agg}
    df = df.rename(columns=rename_back_f2)
    df = df.drop(columns=['fighter_id'], errors='ignore')
    
    # Calculate Diffs
    for c in cols_to_agg:
        c1 = f"f_1_{c}_avg"
        c2 = f"f_2_{c}_avg"
        if c1 in df.columns and c2 in df.columns:
            df[f"diff_{c}_avg"] = df[c1] - df[c2]
            
    return df
