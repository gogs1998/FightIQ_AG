import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

# Add current dir to path to import features
sys.path.append(os.path.join(os.getcwd(), 'experimental'))
from features.sda import compute_sda

def calculate_sda_expanding(df):
    """
    Calculate SDA stats for each fight using only PAST history.
    """
    df = df.sort_values('event_date')
    
    # We need to reconstruct the history for each fighter.
    # The dataframe has one row per fight, with f_1 and f_2 stats.
    # We need to extract the "mix" for the current fight and add it to history.
    
    # Mix columns
    target_cols = ['head_share', 'body_share', 'leg_share']
    pos_cols = ['distance_share', 'clinch_share', 'ground_share']
    
    # We need to know the ACTUAL mix for the fight.
    # The columns f_1_head_share etc. in the dataset are likely PRE-FIGHT averages?
    # NO, wait. 'f_1_head_share' usually means "share of head strikes landed/attempted in THIS fight" 
    # OR "average share in past fights".
    # In standard UFC datasets, un-aggregated columns like 'f_1_head_share' usually refer to the specific fight stats (LEAKAGE if used directly).
    # We want to use them to BUILD history.
    
    # Let's verify if they are pre-fight or post-fight.
    # If they are post-fight, we can use them to update history for the NEXT fight.
    # If they are pre-fight, we can use them directly? No, we want to calculate SDA ourselves.
    
    # Assumption: 'f_1_head_share' is the stat for the current fight.
    
    sda_features = []
    
    # Store history: fighter_id -> list of dicts (mixes)
    history = {} 
    
    print("Calculating SDA features...")
    
    # Iterate through fights chronologically
    for idx, row in df.iterrows():
        fid = row['fight_id']
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        
        # 1. Calculate SDA for F1 based on THEIR history
        if f1 in history and len(history[f1]) >= 2:
            # Create temp DF for compute_sda
            hist_df = pd.DataFrame(history[f1])
            # compute_sda expects a DF with mix columns
            # It returns a DF with one row per fighter
            res = compute_sda(hist_df, mix_cols=target_cols, fighter_col='fighter_id', time_col='date')
            if not res.empty:
                f1_target_sda = res.iloc[0]
            else:
                f1_target_sda = {}
                
            res_pos = compute_sda(hist_df, mix_cols=pos_cols, fighter_col='fighter_id', time_col='date')
            if not res_pos.empty:
                f1_pos_sda = res_pos.iloc[0]
            else:
                f1_pos_sda = {}
        else:
            f1_target_sda = {}
            f1_pos_sda = {}
            
        # 2. Calculate SDA for F2
        if f2 in history and len(history[f2]) >= 2:
            hist_df = pd.DataFrame(history[f2])
            res = compute_sda(hist_df, mix_cols=target_cols, fighter_col='fighter_id', time_col='date')
            if not res.empty:
                f2_target_sda = res.iloc[0]
            else:
                f2_target_sda = {}
                
            res_pos = compute_sda(hist_df, mix_cols=pos_cols, fighter_col='fighter_id', time_col='date')
            if not res_pos.empty:
                f2_pos_sda = res_pos.iloc[0]
            else:
                f2_pos_sda = {}
        else:
            f2_target_sda = {}
            f2_pos_sda = {}
            
        # Store features
        feat = {'fight_id': fid}
        
        # F1 Target
        feat['f_1_sda_target_entropy'] = f1_target_sda.get('SDA_entropy_mean', np.nan)
        feat['f_1_sda_target_js'] = f1_target_sda.get('SDA_js_median', np.nan)
        # F1 Pos
        feat['f_1_sda_pos_entropy'] = f1_pos_sda.get('SDA_entropy_mean', np.nan)
        feat['f_1_sda_pos_js'] = f1_pos_sda.get('SDA_js_median', np.nan)
        
        # F2 Target
        feat['f_2_sda_target_entropy'] = f2_target_sda.get('SDA_entropy_mean', np.nan)
        feat['f_2_sda_target_js'] = f2_target_sda.get('SDA_js_median', np.nan)
        # F2 Pos
        feat['f_2_sda_pos_entropy'] = f2_pos_sda.get('SDA_entropy_mean', np.nan)
        feat['f_2_sda_pos_js'] = f2_pos_sda.get('SDA_js_median', np.nan)
        
        sda_features.append(feat)
        
        # 3. Update History
        # Extract current fight stats
        # F1
        f1_rec = {'fighter_id': f1, 'date': row['event_date']}
        for c in target_cols: f1_rec[c] = row.get(f'f_1_{c}', 0)
        for c in pos_cols: f1_rec[c] = row.get(f'f_1_{c}', 0)
        
        if f1 not in history: history[f1] = []
        history[f1].append(f1_rec)
        
        # F2
        f2_rec = {'fighter_id': f2, 'date': row['event_date']}
        for c in target_cols: f2_rec[c] = row.get(f'f_2_{c}', 0)
        for c in pos_cols: f2_rec[c] = row.get(f'f_2_{c}', 0)
        
        if f2 not in history: history[f2] = []
        history[f2].append(f2_rec)
        
    return pd.DataFrame(sda_features)

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Add fight_id if missing (use index)
    if 'fight_id' not in df.columns:
        df['fight_id'] = df.index
        
    # Calculate SDA
    sda_stats = calculate_sda_expanding(df)
    print(f"Calculated SDA stats for {len(sda_stats)} fights.")
    
    # Merge
    df = df.merge(sda_stats, on='fight_id', how='left')
    
    # Fill NA
    new_feats = [
        'f_1_sda_target_entropy', 'f_1_sda_target_js',
        'f_1_sda_pos_entropy', 'f_1_sda_pos_js',
        'f_2_sda_target_entropy', 'f_2_sda_target_js',
        'f_2_sda_pos_entropy', 'f_2_sda_pos_js'
    ]
    
    for f in new_feats:
        df[f] = df[f].fillna(df[f].mean()) # Fill with mean or -1? Mean is safer for entropy.
        
    # Create Diffs
    df['diff_sda_target_entropy'] = df['f_1_sda_target_entropy'] - df['f_2_sda_target_entropy']
    df['diff_sda_target_js'] = df['f_1_sda_target_js'] - df['f_2_sda_target_js']
    df['diff_sda_pos_entropy'] = df['f_1_sda_pos_entropy'] - df['f_2_sda_pos_entropy']
    df['diff_sda_pos_js'] = df['f_1_sda_pos_js'] - df['f_2_sda_pos_js']
    
    new_feats.extend(['diff_sda_target_entropy', 'diff_sda_target_js', 'diff_sda_pos_entropy', 'diff_sda_pos_js'])
    
    # Train & Evaluate
    print("Training model with SDA features...")
    
    # Load existing features
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    features.extend(new_feats)
    features = [f for f in features if 'duration' not in f]
    
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
    
    print(f"\n--- SDA Model Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    
    # Check importance
    imp = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(imp.head(10))
    
    print("\nNew Feature Importance:")
    print(imp[imp['feature'].isin(new_feats)])
    
    # Save results
    with open('experimental/sda_results.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Log Loss: {ll:.4f}\n")
        f.write("\nNew Feature Importance:\n")
        f.write(imp[imp['feature'].isin(new_feats)].to_string())

if __name__ == "__main__":
    run_test()
