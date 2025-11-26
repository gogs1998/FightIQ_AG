import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import json
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from experimental.features.round_dynamics import build_behavioural_profiles

def prep_rounds(df_golden):
    """
    Prepare round data from golden CSV.
    Need to convert from wide (f_1_r1_sig_str) to long (fighter_id, round, sig_str).
    """
    # This is similar to PEAR prep.
    # We need columns: fighter_id, fight_id, fight_date, round, pace, ctrl_share
    
    # Identify round columns
    # f_1_rX_sig_str_landed, f_1_rX_sig_str_att -> pace = landed / 5 (approx, assuming 5 min rounds)
    # Actually, let's just use landed count as proxy for pace if time unknown.
    
    long_rows = []
    
    for idx, row in df_golden.iterrows():
        # Generate a unique fight_id since it's missing
        date = row['event_date']
        f1_name = row['f_1_name']
        f2_name = row['f_2_name']
        fid = f"{date}_{f1_name}_{f2_name}"
        
        # Max rounds usually 3 or 5. Let's check columns.
        for r in range(1, 6):
            # Fighter 1
            s_land = row.get(f'f_1_r{r}_sig_strikes_succ')
            s_att = row.get(f'f_1_r{r}_sig_strikes_att')
            ctrl = row.get(f'f_1_r{r}_ctrl')
            
            if pd.notna(s_land):
                # Pace: Strikes per minute. Assume 5 min round unless it's the last round and finished early.
                # For simplicity, just use raw count as "Pace Score"
                pace = s_land 
                
                # Ctrl Share: Ctrl Sec / 300
                ctrl_share = ctrl / 300.0 if pd.notna(ctrl) else np.nan
                
                long_rows.append({
                    'fighter_id': f1_name,
                    'fight_id': fid,
                    'fight_date': date,
                    'round': r,
                    'pace': pace,
                    'ctrl_share': ctrl_share
                })
                
            # Fighter 2
            s_land2 = row.get(f'f_2_r{r}_sig_strikes_succ')
            ctrl2 = row.get(f'f_2_r{r}_ctrl')
            
            if pd.notna(s_land2):
                pace2 = s_land2
                ctrl_share2 = ctrl2 / 300.0 if pd.notna(ctrl2) else np.nan
                
                long_rows.append({
                    'fighter_id': f2_name,
                    'fight_id': fid,
                    'fight_date': date,
                    'round': r,
                    'pace': pace2,
                    'ctrl_share': ctrl_share2
                })
                
    return pd.DataFrame(long_rows)

def calculate_expanding_profiles(df_rounds, df_fights):
    """
    Calculate profiles for each fight in df_fights using ONLY rounds from BEFORE that fight.
    """
    df_rounds['fight_date'] = pd.to_datetime(df_rounds['fight_date'])
    df_fights['event_date'] = pd.to_datetime(df_fights['event_date'])
    
    # Sort by date
    df_fights = df_fights.sort_values('event_date')
    
    # We can do this efficiently by iterating through time blocks or just row by row?
    # Row by row is slow.
    # Let's use expanding window by date.
    
    unique_dates = sorted(df_fights['event_date'].unique())
    
    features_list = []
    
    # Pre-group rounds by fighter for faster access?
    # Or just filter.
    
    print(f"Calculating profiles for {len(unique_dates)} dates...")
    
    # Optimization: Calculate all profiles at once? No, leakage.
    # Optimization: Batch updates.
    
    # Let's just do a simple loop with a "past rounds" dataframe that grows.
    
    # Actually, `build_behavioural_profiles` takes a dataframe and returns 1 row per fighter.
    # If we call it every time, it's O(N^2).
    
    # Better approach:
    # 1. Calculate per-fight stats (slope, r1 pace) for ALL fights in history first.
    # 2. Then for each target fight, average the stats of previous fights.
    
    # Step 1: Per-fight stats
    # We can reuse logic from build_behavioural_profiles but grouping by fight_id
    
    per_fight_stats = []
    for (fid, fight_id), g in df_rounds.groupby(['fighter_id', 'fight_id']):
        g = g.sort_values('round')
        r = g['round'].to_numpy()
        pace = g['pace'].to_numpy(float)
        ctrl = g['ctrl_share'].to_numpy(float)
        
        r1_pace = pace[r==1].mean() if (r==1).any() else np.nan
        r3_pace = pace[r==3].mean() if (r==3).any() else np.nan
        ratio = r3_pace / r1_pace if (r1_pace and r1_pace > 0) else np.nan
        
        slope_pace = np.nan
        if len(r) >= 2:
             A = np.column_stack([np.ones(len(r)), r])
             if (~np.isnan(pace)).sum() >= 2:
                 b, *_ = np.linalg.lstsq(A[~np.isnan(pace)], pace[~np.isnan(pace)], rcond=None)
                 slope_pace = b[1]
                 
        per_fight_stats.append({
            'fighter_id': fid,
            'fight_id': fight_id,
            'fight_date': g['fight_date'].iloc[0],
            'stat_r1_pace': r1_pace,
            'stat_ratio': ratio,
            'stat_slope': slope_pace,
            'stat_avg_pace': np.nanmean(pace)
        })
        
    df_stats = pd.DataFrame(per_fight_stats)
    df_stats = df_stats.sort_values('fight_date')
    
    # Step 2: Expanding mean
    # For each fighter, calculate expanding mean/std of these stats.
    # Shift(1) to ensure we only use past fights.
    
    cols_to_agg = ['stat_r1_pace', 'stat_ratio', 'stat_slope', 'stat_avg_pace']
    
    df_stats_expanding = df_stats.groupby('fighter_id')[cols_to_agg].expanding().mean().reset_index(level=0, drop=True)
    # This gives mean including current. We need to shift.
    
    # Actually, easier:
    # Group by fighter
    # Shift columns
    # Then join back to df_fights
    
    final_feats = []
    
    # We need to map these back to df_fights.
    # df_fights has f_1_name, f_2_name, event_date.
    # We need the stats for f_1 known at event_date.
    
    # Let's compute expanding means per fighter
    fighter_history = {} # fid -> DataFrame of stats
    
    for fid, g in df_stats.groupby('fighter_id'):
        g = g.sort_values('fight_date')
        # Expanding mean
        expanded = g[cols_to_agg].expanding().mean()
        # Shift by 1 to exclude current fight
        expanded = expanded.shift(1)
        
        # Combine with date to know when these stats are valid
        # The stats at row i are valid for any fight AFTER row i's date
        # Actually, the stats at row i (shifted) represent average of 0..i-1.
        # They are valid for the fight at row i.
        
        g_feat = pd.concat([g[['fight_date']], expanded], axis=1)
        g_feat.columns = ['fight_date'] + [f"behav_{c}" for c in cols_to_agg]
        fighter_history[fid] = g_feat
        
    # Now merge into df_fights
    # For each row in df_fights, look up f1 and f2 stats.
    # We need the most recent stats BEFORE event_date.
    
    # This lookup can be slow.
    # Optimization: merge_asof
    
    df_fights = df_fights.sort_values('event_date')
    
    # Prepare lookup tables
    lookup_dfs = []
    for fid, hist in fighter_history.items():
        hist['fighter_id'] = fid
        lookup_dfs.append(hist)
        
    if not lookup_dfs:
        return pd.DataFrame()
        
    df_lookup = pd.concat(lookup_dfs).sort_values('fight_date').dropna()
    
    # Merge asof for F1
    df_f1 = pd.merge_asof(
        df_fights.sort_values('event_date'),
        df_lookup.sort_values('fight_date'),
        left_on='event_date',
        right_on='fight_date',
        left_by='f_1_name',
        right_by='fighter_id',
        direction='backward' # Use latest past date
    )
    
    # Rename cols
    rename_map = {c: f"f_1_{c}" for c in df_lookup.columns if c.startswith('behav_')}
    df_f1 = df_f1.rename(columns=rename_map)
    
    # Merge asof for F2
    df_final = pd.merge_asof(
        df_f1.sort_values('event_date'),
        df_lookup.sort_values('fight_date'),
        left_on='event_date',
        right_on='fight_date',
        left_by='f_2_name',
        right_by='fighter_id',
        direction='backward'
    )
    
    rename_map_2 = {c: f"f_2_{c}" for c in df_lookup.columns if c.startswith('behav_')}
    df_final = df_final.rename(columns=rename_map_2)
    
    # Calculate diffs
    for c in cols_to_agg:
        base = f"behav_{c}"
        if f"f_1_{base}" in df_final.columns and f"f_2_{base}" in df_final.columns:
            df_final[f"diff_{base}"] = df_final[f"f_1_{base}"] - df_final[f"f_2_{base}"]
            
    return df_final

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df_golden = pd.read_csv('UFC_full_data_golden.csv')
    
    print("Preparing round data...")
    df_rounds = prep_rounds(df_golden)
    print(f"Extracted {len(df_rounds)} round records.")
    
    print("Calculating Behavioural Profiles (Expanding Window)...")
    df_aug = calculate_expanding_profiles(df_rounds, df)
    
    # Check new columns
    new_cols = [c for c in df_aug.columns if 'behav_' in c]
    print(f"Generated {len(new_cols)} behavioural features.")
    print(f"Sample features: {new_cols[:5]}")
    
    # Prepare for XGBoost
    with open('features_elo.json', 'r') as f:
        base_features = json.load(f)
        
    # Add new features (use diffs)
    diff_cols = [c for c in new_cols if c.startswith('diff_')]
    all_features = base_features + diff_cols
    
    print(f"Total Features: {len(all_features)}")
    
    # Encode
    cat_cols = df_aug[base_features].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_aug[col] = df_aug[col].astype(str)
        df_aug[col] = le.fit_transform(df_aug[col])
        
    # Target
    f1_wins = df_aug['winner'] == df_aug['f_1_name']
    f2_wins = df_aug['winner'] == df_aug['f_2_name']
    df_aug = df_aug[f1_wins | f2_wins].copy()
    df_aug['target'] = (df_aug['winner'] == df_aug['f_1_name']).astype(int)
    
    # Split
    split_idx = int(len(df_aug) * 0.85)
    train_df = df_aug.iloc[:split_idx]
    test_df = df_aug.iloc[split_idx:]
    
    X_train = train_df[all_features]
    y_train = train_df['target']
    X_test = test_df[all_features]
    y_test = test_df['target']
    
    # Baseline (only base features)
    print("\nTraining Baseline XGBoost...")
    xgb_base = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_base.fit(train_df[base_features], y_train)
    p_base = xgb_base.predict_proba(test_df[base_features])[:, 1]
    acc_base = accuracy_score(y_test, (p_base > 0.5).astype(int))
    ll_base = log_loss(y_test, p_base)
    print(f"Baseline: Acc {acc_base:.4f}, LL {ll_base:.4f}")
    
    # Experimental (with behavioural features)
    print("\nTraining Experimental XGBoost (with Behavioural Profiles)...")
    xgb_exp = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_exp.fit(X_train, y_train)
    p_exp = xgb_exp.predict_proba(X_test)[:, 1]
    acc_exp = accuracy_score(y_test, (p_exp > 0.5).astype(int))
    ll_exp = log_loss(y_test, p_exp)
    print(f"Experimental: Acc {acc_exp:.4f}, LL {ll_exp:.4f}")
    
    # Save results
    with open('experimental/BEHAVIOUR_RESULTS.md', 'w') as f:
        f.write(f"# Behavioural Profiles Results\n\n")
        f.write(f"## Metrics\n")
        f.write(f"| Model | Accuracy | Log Loss |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| Baseline | {acc_base:.4f} | {ll_base:.4f} |\n")
        f.write(f"| Experimental | {acc_exp:.4f} | {ll_exp:.4f} |\n\n")
        
        # Feature Importance
        imp = pd.Series(xgb_exp.feature_importances_, index=all_features).sort_values(ascending=False)
        f.write(f"## Top 10 Features\n")
        for name, val in imp.head(10).items():
            f.write(f"- {name}: {val:.4f}\n")
            
        # Check specific behavioural features
        f.write(f"\n## Behavioural Feature Importance\n")
        behav_imp = imp[imp.index.isin(diff_cols)]
        for name, val in behav_imp.sort_values(ascending=False).items():
            f.write(f"- {name}: {val:.4f}\n")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
