import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def backtest_afm_roi():
    print("=== AFM ROI Backtest (2024-2025) ===")
    
    # 1. Load Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    # Load Boruta Features
    with open(f'{BASE_DIR}/experiment_2/boruta_results.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    # Filter valid odds & time
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    # 2. Compute AFM (Quickly re-compute to ensure we have it)
    print("Computing AFM...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    mask_train_surr = df['event_date'] < '2024-01-01'
    surrogate = LogisticRegression(C=1.0, max_iter=1000)
    surrogate.fit(X_scaled[mask_train_surr], y[mask_train_surr])
    
    base_probs = surrogate.predict_proba(X_scaled)[:, 1]
    
    # Simple perturbation (10 iterations for speed)
    min_probs = base_probs.copy()
    max_probs = base_probs.copy()
    
    for i in range(10):
        noise = np.random.normal(0, 0.2, X_scaled.shape)
        probs_noisy = surrogate.predict_proba(X_scaled + noise)[:, 1]
        min_probs = np.minimum(min_probs, probs_noisy)
        max_probs = np.maximum(max_probs, probs_noisy)
        
    df['afm_fragile'] = ((base_probs < 0.5) & (max_probs > 0.5)) | \
                        ((base_probs > 0.5) & (min_probs < 0.5))
    df['afm_fragile'] = df['afm_fragile'].astype(int)
    
    # 3. Train Main Model (Without AFM features, to keep original accuracy)
    # We want to see if filtering the *original* model's bets with AFM helps.
    # Or should we use the AFM-trained model?
    # User asked: "test roi with afm and lower accuracy" -> implies using the AFM-trained model.
    # BUT, usually filtering is better. Let's try BOTH.
    
    # Scenario A: Use AFM-Trained Model (Lower Accuracy)
    print("Training AFM-Inclusive Model...")
    afm_feats = ['afm_fragile'] # Just use the flag for simplicity
    X_afm = df[features + afm_feats].fillna(0)
    
    mask_train = df['event_date'] < '2024-01-01'
    mask_test = df['event_date'] >= '2024-01-01'
    
    model_afm = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
    model_afm.fit(X_afm[mask_train], y[mask_train])
    probs_afm = model_afm.predict_proba(X_afm[mask_test])[:, 1]
    
    # Scenario B: Use Original Boruta Model (Higher Accuracy) but Filter Fragile Bets
    print("Training Original Boruta Model...")
    model_orig = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
    model_orig.fit(X[mask_train], y[mask_train])
    probs_orig = model_orig.predict_proba(X[mask_test])[:, 1]
    
    # 4. Backtest
    test_df = df[mask_test].copy()
    test_df['prob_afm'] = probs_afm
    test_df['prob_orig'] = probs_orig
    
    strategies = {
        "1. Original (Value Sniper)": [],
        "2. AFM-Trained Model (Value Sniper)": [],
        "3. Original + Filter Fragile": []
    }
    
    bankrolls = {k: 1000.0 for k in strategies}
    
    for idx, row in test_df.iterrows():
        # 1. Original
        p = row['prob_orig']
        if p > 0.5:
            my_prob = p
            odds = row['f_1_odds']
            win = (row['target'] == 1)
        else:
            my_prob = 1 - p
            odds = row['f_2_odds']
            win = (row['target'] == 0)
            
        edge = my_prob - (1/odds)
        if edge > 0.05:
            stake = 50
            res = (stake * (odds - 1)) if win else -stake
            bankrolls["1. Original (Value Sniper)"] += res
            
            # 3. Original + Filter
            # Only bet if NOT fragile
            if row['afm_fragile'] == 0:
                bankrolls["3. Original + Filter Fragile"] += res
                
        # 2. AFM-Trained
        p_afm = row['prob_afm']
        if p_afm > 0.5:
            my_prob = p_afm
            odds = row['f_1_odds']
            win = (row['target'] == 1)
        else:
            my_prob = 1 - p_afm
            odds = row['f_2_odds']
            win = (row['target'] == 0)
            
        edge = my_prob - (1/odds)
        if edge > 0.05:
            stake = 50
            res = (stake * (odds - 1)) if win else -stake
            bankrolls["2. AFM-Trained Model (Value Sniper)"] += res
            
    print("\n=== ROI Results (2024-2025) ===")
    print(f"{'Strategy':<40} | {'End Bank':<12} | {'ROI':<8}")
    print("-" * 65)
    
    for k, v in bankrolls.items():
        roi = (v - 1000) / 1000
        print(f"{k:<40} | ${v:,.2f}   | {roi:<8.1%}")

if __name__ == "__main__":
    backtest_afm_roi()
