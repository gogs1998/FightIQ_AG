import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import accuracy_score

def check_2025_roi_afm():
    print("=== FightIQ: 2025 ROI Check (AFM Edition) ===")
    
    # 1. Load Data (WITH AFM)
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data_with_afm.csv')
    except:
        print("Error: AFM dataset not found.")
        return
        
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # 2. Load Features & Params
    with open(f'{BASE_DIR}/prop_hunter/features.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    # Add AFM Features
    afm_features = ['afm_upside', 'afm_downside', 'afm_skew', 'afm_fragile']
    all_features = features + afm_features
    
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
        
    # 3. Split: Train (2010-2024), Test (2025)
    train_mask = (df['event_date'] >= '2010-01-01') & (df['event_date'] < '2025-01-01')
    test_mask = (df['event_date'] >= '2025-01-01')
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    print(f"Training on {len(train_df)} fights (2010-2024)...")
    print(f"Testing on {len(test_df)} fights (2025)...")
    
    X_train = train_df[all_features].fillna(0)
    y_train = train_df['target']
    
    X_test = test_df[all_features].fillna(0)
    y_test = test_df['target']
    
    # 4. Train & Predict
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy (2025): {acc:.1%}")
    
    # 5. Calculate Sniper ROI
    profit = 0
    invested = 0
    
    for i, idx in enumerate(test_df.index):
        row = test_df.loc[idx]
        prob = probs[i]
        target = y_test.iloc[i]
        
        odds_1 = row['f_1_odds']
        odds_2 = row['f_2_odds']
        
        if pd.isna(odds_1) or pd.isna(odds_2): continue
        
        edge = 0
        if prob > 0.5:
            implied = 1/odds_1
            if (prob - implied) > 0.05:
                bet_on = 1
                odds = odds_1
                edge = prob - implied
        else:
            implied = 1/odds_2
            if ((1-prob) - implied) > 0.05:
                bet_on = 0
                odds = odds_2
                edge = (1-prob) - implied
                
        if edge > 0:
            invested += 1
            if bet_on == target:
                profit += (odds - 1)
            else:
                profit -= 1
                
    roi = profit / invested if invested > 0 else 0
    print(f"Sniper ROI (2025): {roi:.1%} ({invested} bets)")

if __name__ == "__main__":
    check_2025_roi_afm()
