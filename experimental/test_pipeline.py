import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

# Add current dir to path to import features
sys.path.append(os.path.join(os.getcwd(), 'experimental'))
from features.pear import fit_pear

def prep_rounds(df):
    """
    Convert wide UFC data to long round-by-round format for PEAR.
    """
    rounds = []
    
    # Iterate through fights
    for idx, row in df.iterrows():
        fid = row.get('fight_id', idx) # Use index if no fight_id
        date = row['event_date']
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        
        # We assume 5 rounds max
        for r in range(1, 6):
            # Check if round happened (if stats are not null)
            # We use 'f_1_r{r}_sig_strikes' as proxy
            col_landed = f'f_1_r{r}_sig_strikes'
            if col_landed not in df.columns or pd.isna(row[col_landed]):
                continue
                
            # Extract stats
            # PEAR needs: 'sig_str_diff', 'opp_sig_str_per_min'
            # We approximate 5 min rounds
            
            s1 = row.get(f'f_1_r{r}_sig_strikes', 0)
            s2 = row.get(f'f_2_r{r}_sig_strikes', 0)
            
            # F1 perspective
            rounds.append({
                'fight_id': fid,
                'fighter_id': f1,
                'opponent_id': f2,
                'date': date,
                'round': r,
                'sig_str_diff': s1 - s2,
                'opp_sig_str_per_min': s2 / 5.0
            })
            
            # F2 perspective
            rounds.append({
                'fight_id': fid,
                'fighter_id': f2,
                'opponent_id': f1,
                'date': date,
                'round': r,
                'sig_str_diff': s2 - s1,
                'opp_sig_str_per_min': s1 / 5.0
            })
            
    return pd.DataFrame(rounds)

def calculate_pear_expanding(df_rounds):
    """
    Calculate PEAR stats for each fight using only PAST history.
    Returns a DataFrame with ['fight_id', 'f_1_beta_pace', 'f_1_beta_lag', 'f_2_beta_pace', 'f_2_beta_lag']
    """
    df_rounds = df_rounds.sort_values('date')
    
    # Store history per fighter
    fighter_history = {} # fighter_id -> list of round rows
    
    results = []
    
    # Group by fight_id to process fights chronologically
    # Note: df_rounds has 2 rows per round (one per fighter). 
    # We need to process unique fights.
    # unique_fights = df_rounds[['fight_id', 'date', 'fighter_id', 'opponent_id']].drop_duplicates()
    
    # This is tricky because unique_fights has 2 rows per fight (one per fighter perspective).
    # Let's just iterate through the main df (UFC_data) and query the rounds.
    # Or better: Iterate through unique dates.
    
    # Optimization: We only need to calculate this for the rows in our training/test set.
    # But we need to build history from the beginning.
    
    # Let's do this:
    # 1. Group rounds by fighter.
    # 2. For each fighter, iterate through their fights.
    # 3. At each fight, fit PEAR on *previous* fights.
    
    print("Calculating PEAR features (this may take a moment)...")
    print(f"df_rounds columns: {df_rounds.columns.tolist()}")
    
    # DEBUG: Check if fight_id is in columns
    if 'fight_id' not in df_rounds.columns:
        print("CRITICAL ERROR: fight_id NOT in df_rounds columns!")
        return pd.DataFrame()
    
    pear_features = []
    
    # Get all unique fighters
    fighters = df_rounds['fighter_id'].unique()
    
    for f in fighters:
        f_rounds = df_rounds[df_rounds['fighter_id'] == f].sort_values(['date', 'round'])
        
        # Get unique fights for this fighter
        f_fights = f_rounds['fight_id'].unique()
        
        # We need to calculate the PEAR stats *entering* each fight.
        # So for fight i, we use rounds from fights 0 to i-1.
        
        # Accumulate history
        history_rounds = []
        
        for i, fid in enumerate(f_fights):
            # Current fight rounds
            curr_rounds = f_rounds[f_rounds['fight_id'] == fid]
            
            # Calculate stats based on history (if enough history)
            if len(history_rounds) >= 5: # Min rounds to fit
                hist_df = pd.DataFrame(history_rounds)
                
                # Fit PEAR
                # We can reuse fit_pear but it expects a DF with multiple fighters. 
                # We pass just this fighter's history.
                # fit_pear returns a DF with one row.
                res = fit_pear(hist_df)
                
                if not res.empty:
                    stats = res.iloc[0]
                    pear_features.append({
                        'fight_id': fid,
                        'fighter_id': f,
                        'beta_pace': stats['beta_pace'],
                        'beta_lag': stats['beta_lag']
                    })
            
            # Add current rounds to history for NEXT fight
            history_rounds.extend(curr_rounds.to_dict('records'))
            
    return pd.DataFrame(pear_features)

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Add fight_id if missing (use index)
    if 'fight_id' not in df.columns:
        df['fight_id'] = df.index
    
    # 1. Prep Rounds
    print("Prepping round data...")
    df_rounds = prep_rounds(df)
    print(f"Generated {len(df_rounds)} round records.")
    
    # 2. Calculate PEAR
    try:
        pear_stats = calculate_pear_expanding(df_rounds)
        print(f"Calculated PEAR stats for {len(pear_stats)} fighter-fight instances.")
    except Exception as e:
        print(f"ERROR in calculate_pear_expanding: {e}")
        print(f"df_rounds columns: {df_rounds.columns.tolist()}")
        if not df_rounds.empty:
            print("First row:", df_rounds.iloc[0].to_dict())
        import traceback
        traceback.print_exc()
        return
    
    # 3. Merge back to main DF
    # pear_stats has columns: ['fight_id', 'fighter_id', 'beta_pace', 'beta_lag']
    # We need to merge on fight_id and fighter_id
    
    # Ensure columns exist
    if 'fight_id' not in pear_stats.columns:
        print("ERROR: fight_id missing from pear_stats")
        print(pear_stats.columns)
        return

    # Merge for F1
    # We merge on fight_id and f_1_name matching fighter_id
    df = df.merge(pear_stats, left_on=['fight_id', 'f_1_name'], right_on=['fight_id', 'fighter_id'], how='left')
    df.rename(columns={'beta_pace': 'f_1_beta_pace', 'beta_lag': 'f_1_beta_lag'}, inplace=True)
    df.drop(columns=['fighter_id'], inplace=True)
    
    # Merge for F2
    df = df.merge(pear_stats, left_on=['fight_id', 'f_2_name'], right_on=['fight_id', 'fighter_id'], how='left')
    df.rename(columns={'beta_pace': 'f_2_beta_pace', 'beta_lag': 'f_2_beta_lag'}, inplace=True)
    df.drop(columns=['fighter_id'], inplace=True)
    
    # Fill NA (first fights or not enough history) with 0 or mean
    df['f_1_beta_pace'] = df['f_1_beta_pace'].fillna(0)
    df['f_1_beta_lag'] = df['f_1_beta_lag'].fillna(0)
    df['f_2_beta_pace'] = df['f_2_beta_pace'].fillna(0)
    df['f_2_beta_lag'] = df['f_2_beta_lag'].fillna(0)
    
    # Create Diff features
    df['diff_beta_pace'] = df['f_1_beta_pace'] - df['f_2_beta_pace']
    df['diff_beta_lag'] = df['f_1_beta_lag'] - df['f_2_beta_lag']
    
    # 4. Train & Evaluate
    print("Training model with PEAR features...")
    
    # Load existing features
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    # Add new features
    new_feats = ['f_1_beta_pace', 'f_1_beta_lag', 'f_2_beta_pace', 'f_2_beta_lag', 'diff_beta_pace', 'diff_beta_lag']
    features.extend(new_feats)
    
    # Filter leakage (durations) just in case
    features = [f for f in features if 'duration' not in f]
    
    # Prepare Train/Test
    # ... (Copy training logic from train_elo.py) ...
    # For brevity, I'll just use the same split logic
    
    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    # Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']
    
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        colsample_bytree=0.6,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    from sklearn.metrics import accuracy_score, log_loss
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    print(f"\n--- PEAR Model Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    
    # Check importance of new features
    imp = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(imp.head(10))
    
    print("\nNew Feature Importance:")
    print(imp[imp['feature'].isin(new_feats)])

if __name__ == "__main__":
    run_test()
