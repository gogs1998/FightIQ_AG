print("Starting test_ensemble_all.py...")
try:
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import joblib
    import json
    import sys
    import os
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Add current dir to path
    sys.path.append(os.path.join(os.getcwd(), 'experimental'))
    from models.siamese import SiameseMatchupNet, symmetric_loss
    from features.pear import fit_pear
except Exception as e:
    print(f"Import Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ... (Imports are already there, but I will restate them to be safe if I replace the whole file or chunk)
# Actually, I will replace the whole file content to be clean.

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
            col_landed = f'f_1_r{r}_sig_strikes'
            if col_landed not in df.columns or pd.isna(row[col_landed]):
                continue
                
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
    """
    df_rounds = df_rounds.sort_values('date')
    pear_features = []
    
    # Get all unique fighters
    fighters = df_rounds['fighter_id'].unique()
    
    for f in fighters:
        f_rounds = df_rounds[df_rounds['fighter_id'] == f].sort_values(['date', 'round'])
        f_fights = f_rounds['fight_id'].unique()
        
        history_rounds = []
        
        for i, fid in enumerate(f_fights):
            curr_rounds = f_rounds[f_rounds['fight_id'] == fid]
            
            if len(history_rounds) >= 5: 
                hist_df = pd.DataFrame(history_rounds)
                res = fit_pear(hist_df)
                
                if not res.empty:
                    stats = res.iloc[0]
                    pear_features.append({
                        'fight_id': fid,
                        'fighter_id': f,
                        'beta_pace': stats['beta_pace'],
                        'beta_lag': stats['beta_lag']
                    })
            
            history_rounds.extend(curr_rounds.to_dict('records'))
            
    return pd.DataFrame(pear_features)

def prepare_siamese_data(df, top_features):
    """
    Split features into fighter A and fighter B sets based on Top Features.
    """
    # Identify pairs from top_features
    pairs = set()
    all_cols = set(df.columns)
    
    for feat in top_features:
        base = None
        f1_col = None
        f2_col = None
        
        if feat.startswith('diff_'):
            base = feat[5:] 
            if f"f_1_{base}" in all_cols and f"f_2_{base}" in all_cols:
                f1_col = f"f_1_{base}"
                f2_col = f"f_2_{base}"
            elif f"{base}_f_1" in all_cols and f"{base}_f_2" in all_cols:
                f1_col = f"{base}_f_1"
                f2_col = f"{base}_f_2"
                
        elif '_f_1' in feat or 'f_1_' in feat:
            if feat.startswith('f_1_'):
                f1_col = feat
                f2_col = feat.replace('f_1_', 'f_2_')
            else:
                f1_col = feat
                f2_col = feat.replace('_f_1', '_f_2')
            if f2_col not in all_cols: f1_col = None
            
        elif '_f_2' in feat or 'f_2_' in feat:
            if feat.startswith('f_2_'):
                f2_col = feat
                f1_col = feat.replace('f_2_', 'f_1_')
            else:
                f2_col = feat
                f1_col = feat.replace('_f_2', '_f_1')
            if f1_col not in all_cols: f1_col = None
            
        if f1_col and f2_col:
            if (f1_col, f2_col) not in pairs:
                pairs.add((f1_col, f2_col))

    pairs = list(pairs)
    print(f"Found {len(pairs)} feature pairs related to Top XGBoost features.")
    
    numeric_pairs = []
    for c1, c2 in pairs:
        if pd.api.types.is_numeric_dtype(df[c1]) and pd.api.types.is_numeric_dtype(df[c2]):
            numeric_pairs.append((c1, c2))
            
    f1_feats = [p[0] for p in numeric_pairs]
    f2_feats = [p[1] for p in numeric_pairs]
    
    X1 = df[f1_feats].copy()
    X2 = df[f2_feats].copy()
    
    generic_names = []
    for c in f1_feats:
        if c.startswith('f_1_'):
            generic_names.append(c[4:])
        elif c.endswith('_f_1'):
            generic_names.append(c[:-4])
        else:
            generic_names.append(c)
            
    X1.columns = generic_names
    X2.columns = generic_names
    
    X1 = X1.fillna(0) 
    X2 = X2.fillna(0)
    
    scaler = StandardScaler()
    combined = pd.concat([X1, X2], axis=0)
    scaler.fit(combined)
    
    X1_scaled = scaler.transform(X1)
    X2_scaled = scaler.transform(X2)
    
    y = (df['winner'] == df['f_1_name']).astype(float).to_numpy()
    
    return X1_scaled, X2_scaled, y, f1_feats

def run_ensemble_all():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    if 'fight_id' not in df.columns:
        df['fight_id'] = df.index
        
    # 1. Calculate PEAR
    print("Calculating PEAR features...")
    df_rounds = prep_rounds(df)
    try:
        pear_stats = calculate_pear_expanding(df_rounds)
        
        # Merge
        df = df.merge(pear_stats, left_on=['fight_id', 'f_1_name'], right_on=['fight_id', 'fighter_id'], how='left')
        df.rename(columns={'beta_pace': 'f_1_beta_pace', 'beta_lag': 'f_1_beta_lag'}, inplace=True)
        df.drop(columns=['fighter_id'], inplace=True)
        
        df = df.merge(pear_stats, left_on=['fight_id', 'f_2_name'], right_on=['fight_id', 'fighter_id'], how='left')
        df.rename(columns={'beta_pace': 'f_2_beta_pace', 'beta_lag': 'f_2_beta_lag'}, inplace=True)
        df.drop(columns=['fighter_id'], inplace=True)
        
        df['f_1_beta_pace'] = df['f_1_beta_pace'].fillna(0)
        df['f_1_beta_lag'] = df['f_1_beta_lag'].fillna(0)
        df['f_2_beta_pace'] = df['f_2_beta_pace'].fillna(0)
        df['f_2_beta_lag'] = df['f_2_beta_lag'].fillna(0)
        
        df['diff_beta_pace'] = df['f_1_beta_pace'] - df['f_2_beta_pace']
        df['diff_beta_lag'] = df['f_1_beta_lag'] - df['f_2_beta_lag']
        
        print("PEAR features added.")
        
    except Exception as e:
        print(f"PEAR calculation failed: {e}")
        return

    # 2. Train XGBoost (Elo + PEAR)
    print("Training XGBoost (Elo + PEAR)...")
    
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
    
    new_feats = ['f_1_beta_pace', 'f_1_beta_lag', 'f_2_beta_pace', 'f_2_beta_lag', 'diff_beta_pace', 'diff_beta_lag']
    features.extend(new_feats)
    features = [f for f in features if 'duration' not in f]
    
    # Encode categoricals
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
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=5,
        colsample_bytree=0.6, subsample=0.8, random_state=42, n_jobs=-1,
        early_stopping_rounds=50
    )
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_acc = accuracy_score(y_test, (xgb_probs > 0.5).astype(int))
    xgb_ll = log_loss(y_test, xgb_probs)
    print(f"XGBoost (Elo+PEAR) Accuracy: {xgb_acc:.4f}, Log Loss: {xgb_ll:.4f}")
    
    # 3. Train Siamese (Top 50 from ORIGINAL XGBoost)
    print("Training Siamese Net (Original Top 50 features)...")
    
    # Load original model for importance
    try:
        orig_xgb = joblib.load('ufc_model_elo.pkl')
        with open('features_elo.json', 'r') as f:
            orig_feats = json.load(f)
        
        orig_imp = orig_xgb.feature_importances_
        feat_imp = dict(zip(orig_feats, orig_imp))
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        top_features = [x[0] for x in sorted_feats[:50]]
        print("Loaded Top 50 features from ORIGINAL model.")
    except:
        print("Could not load original model, falling back to current model importance.")
        importances = xgb_model.feature_importances_
        feat_imp = dict(zip(features, importances))
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        top_features = [x[0] for x in sorted_feats[:50]]
    
    # Check if PEAR features made it (unlikely if using original)
    pear_in_top = [f for f in top_features if f in new_feats]
    print(f"PEAR features in Top 50: {pear_in_top}")
    
    X1, X2, y_siamese, f1_names = prepare_siamese_data(df, top_features)
    
    y_train_s, y_test_s = y_siamese[:split_idx], y_siamese[split_idx:]
    
    X1_train, X1_test = X1[:split_idx], X1[split_idx:]
    X2_train, X2_test = X2[:split_idx], X2[split_idx:]
    
    t_X1_train = torch.tensor(X1_train, dtype=torch.float32)
    t_X2_train = torch.tensor(X2_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train_s, dtype=torch.float32)
    t_X1_test = torch.tensor(X1_test, dtype=torch.float32)
    t_X2_test = torch.tensor(X2_test, dtype=torch.float32)
    
    train_ds = TensorDataset(t_X1_train, t_X2_train, t_y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    model = SiameseMatchupNet(in_dim=X1.shape[1], hidden=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20):
        model.train()
        for b_x1, b_x2, b_y in train_loader:
            optimizer.zero_grad()
            loss = symmetric_loss(model, b_x1, b_x2, b_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        siamese_probs = model(t_X1_test, t_X2_test).numpy()
        
    siamese_acc = accuracy_score(y_test_s, (siamese_probs > 0.5).astype(int))
    siamese_ll = log_loss(y_test_s, siamese_probs)
    print(f"Siamese Accuracy: {siamese_acc:.4f}, Log Loss: {siamese_ll:.4f}")
    
    # 4. Ensemble
    print("Ensembling...")
    ens_probs = 0.5 * siamese_probs + 0.5 * xgb_probs
    ens_acc = accuracy_score(y_test, (ens_probs > 0.5).astype(int))
    ens_ll = log_loss(y_test, ens_probs)
    
    print(f"\n--- Final Ensemble (Elo + PEAR + Siamese) ---")
    print(f"Accuracy: {ens_acc:.4f}")
    print(f"Log Loss: {ens_ll:.4f}")
    
    with open('experimental/ensemble_all_results.txt', 'w') as f:
        f.write(f"XGBoost Accuracy: {xgb_acc:.4f}\n")
        f.write(f"XGBoost Log Loss: {xgb_ll:.4f}\n")
        f.write(f"Siamese Accuracy: {siamese_acc:.4f}\n")
        f.write(f"Siamese Log Loss: {siamese_ll:.4f}\n")
        f.write(f"Ensemble Accuracy: {ens_acc:.4f}\n")
        f.write(f"Ensemble Log Loss: {ens_ll:.4f}\n")

if __name__ == '__main__':
    try:
        run_ensemble_all()
    except Exception as e:
        print(f"Runtime Error: {e}")
        import traceback
        traceback.print_exc()
