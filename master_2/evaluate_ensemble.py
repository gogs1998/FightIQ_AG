import pandas as pd
import numpy as np
import joblib
import json
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from models import SiameseMatchupNet, prepare_siamese_data

def evaluate_ensemble():
    print("=== FightIQ: Final Ensemble Evaluation (2024-2025) ===")
    
    BASE_DIR = 'd:/AntiGravity/FightIQ/master_2'
    
    # 1. Load Data
    df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    with open(f'{BASE_DIR}/features.json', 'r') as f: features = json.load(f)
    with open(f'{BASE_DIR}/params.json', 'r') as f: params = json.load(f)['best_params']
    
    # Filter Odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X_df = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    # Split
    split_date = '2024-01-01'
    mask_test = df['event_date'] >= split_date
    
    X_test = X_df[mask_test]
    y_test = y[mask_test]
    test_df = df[mask_test].copy()
    
    print(f"Testing on {len(X_test)} fights (2024-2025).")
    
    # 2. Load Models
    print("Loading Models...")
    xgb_model = joblib.load(f'{BASE_DIR}/models/xgb_optimized.pkl')
    
    # Siamese
    f1_test, f2_test, input_dim, _ = prepare_siamese_data(X_test, features)
    scaler = joblib.load(f'{BASE_DIR}/models/siamese_scaler.pkl')
    
    f1_test = scaler.transform(f1_test)
    f2_test = scaler.transform(f2_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
    siamese_model.load_state_dict(torch.load(f'{BASE_DIR}/models/siamese_optimized.pth'))
    siamese_model.eval()
    
    # 3. Predict
    print("Generating Predictions...")
    
    # XGB
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # Siamese
    with torch.no_grad():
        t1 = torch.FloatTensor(f1_test).to(device)
        t2 = torch.FloatTensor(f2_test).to(device)
        p_siam = siamese_model(t1, t2).cpu().numpy().flatten()
        
    # Ensemble
    w = params['ensemble_xgb_weight']
    p_ens = w * p_xgb + (1 - w) * p_siam
    
    # 4. Metrics
    acc_xgb = accuracy_score(y_test, (p_xgb > 0.5).astype(int))
    acc_siam = accuracy_score(y_test, (p_siam > 0.5).astype(int))
    acc_ens = accuracy_score(y_test, (p_ens > 0.5).astype(int))
    
    print(f"\n--- Accuracy ---")
    print(f"XGBoost: {acc_xgb:.4%}")
    print(f"Siamese: {acc_siam:.4%}")
    print(f"Ensemble: {acc_ens:.4%} 🚀")
    
    # 5. ROI Check (Value Sniper)
    profit = 0
    invested = 0
    
    for i, idx in enumerate(test_df.index):
        row = test_df.loc[idx]
        prob = p_ens[i]
        target = y_test[i]
        odds_1, odds_2 = row['f_1_odds'], row['f_2_odds']
        
        if prob > 0.5:
            implied = 1/odds_1
            if (prob - implied) > 0.05:
                if target == 1: profit += (odds_1 - 1)
                else: profit -= 1
                invested += 1
        else:
            implied = 1/odds_2
            if ((1-prob) - implied) > 0.05:
                if target == 0: profit += (odds_2 - 1)
                else: profit -= 1
                invested += 1
                
    roi = profit / invested if invested > 0 else 0
    print(f"\n--- ROI (Value Sniper) ---")
    print(f"Bets: {invested}")
    print(f"ROI: {roi:.2%}")

if __name__ == "__main__":
    evaluate_ensemble()
