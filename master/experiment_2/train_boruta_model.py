import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

def train_boruta_model():
    print("=== Training FightIQ Boruta Model (Lean) ===")
    
    # 1. Load Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    # Load Boruta Features
    with open(f'{BASE_DIR}/experiment_2/boruta_results.json', 'r') as f:
        boruta_res = json.load(f)
        features = boruta_res['confirmed']
        
    print(f"Using {len(features)} Boruta-confirmed features.")
    
    # Filter valid odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    # 2. Time-Based Split (Same as Main Pipeline)
    split_date = '2024-01-01'
    mask_train = df['event_date'] < split_date
    mask_test = df['event_date'] >= split_date
    
    X_train = X[mask_train]
    X_test = X[mask_test]
    y_train = y[mask_train]
    y_test = y[mask_test]
    
    print(f"Train Set: {len(X_train)} (Pre-2024)")
    print(f"Test Set:  {len(X_test)} (2024-2025)")
    
    # 3. Train XGBoost
    # Using standard params, maybe slightly less regularization needed since we removed noise
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    print(f"\n=== Boruta Model Performance (2024-2025 Holdout) ===")
    print(f"Accuracy: {acc:.4%}")
    print(f"Log Loss: {ll:.4f}")
    
    # 5. ROI Check (Simple Flat Bet)
    # Filter for bets with >60% confidence
    test_df = df[mask_test].copy()
    test_df['prob'] = probs
    test_df['pred'] = preds
    
    roi_bets = []
    
    for idx, row in test_df.iterrows():
        p = row['prob']
        if p > 0.60:
            bet_on_f1 = True
            odds = row['f_1_odds']
            win = (row['target'] == 1)
        elif p < 0.40:
            bet_on_f1 = False
            odds = row['f_2_odds']
            win = (row['target'] == 0)
        else:
            continue
            
        if odds < 1.01: continue
        
        profit = (odds - 1) if win else -1
        roi_bets.append(profit)
        
    if roi_bets:
        roi = sum(roi_bets) / len(roi_bets)
        print(f"\nROI (Flat Bet, Conf > 60%): {roi:.2%}")
        print(f"Total Bets: {len(roi_bets)}")
    else:
        print("\nNo bets met the 60% confidence threshold.")

if __name__ == "__main__":
    train_boruta_model()
