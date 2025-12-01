import pandas as pd
import numpy as np
import joblib
import json
import torch
import optuna
from models import SiameseMatchupNet, prepare_siamese_data

def optimize_weight_roi():
    print("=== Optimizing Ensemble Weight for ROI (Value Sniper) with Optuna ===")
    
    BASE_DIR = 'd:/AntiGravity/FightIQ/master_2'
    
    # Load Data
    df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    with open(f'{BASE_DIR}/features.json', 'r') as f: features = json.load(f)
    with open(f'{BASE_DIR}/params.json', 'r') as f: params = json.load(f)['best_params']
    
    # Filter Odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Split
    mask_test = df['event_date'] >= '2024-01-01'
    X_df = df[[c for c in features if c in df.columns]].fillna(0)
    X_test = X_df[mask_test]
    y_test = df.loc[mask_test, 'target'].values
    test_df = df[mask_test].copy()
    
    # Load Models
    xgb_model = joblib.load(f'{BASE_DIR}/models/xgb_optimized.pkl')
    scaler = joblib.load(f'{BASE_DIR}/models/siamese_scaler.pkl')
    
    f1_test, f2_test, input_dim, _ = prepare_siamese_data(X_test, features)
    f1_test = scaler.transform(f1_test)
    f2_test = scaler.transform(f2_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
    siamese_model.load_state_dict(torch.load(f'{BASE_DIR}/models/siamese_optimized.pth'))
    siamese_model.eval()
    
    # Predict
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    with torch.no_grad():
        t1 = torch.FloatTensor(f1_test).to(device)
        t2 = torch.FloatTensor(f2_test).to(device)
        p_siam = siamese_model(t1, t2).cpu().numpy().flatten()
        
    def objective(trial):
        w = trial.suggest_float('w', 0.0, 1.0)
        p_ens = w * p_xgb + (1 - w) * p_siam
        
        # Calculate ROI
        profit = 0
        invested = 0
        
        for i, idx in enumerate(test_df.index):
            prob = p_ens[i]
            target = y_test[i]
            odds_1, odds_2 = test_df.iloc[i]['f_1_odds'], test_df.iloc[i]['f_2_odds']
            
            if prob > 0.5:
                implied = 1/odds_1
                if (prob - implied) > 0.05: # 5% Edge
                    if target == 1: profit += (odds_1 - 1)
                    else: profit -= 1
                    invested += 1
            else:
                implied = 1/odds_2
                if ((1-prob) - implied) > 0.05: # 5% Edge
                    if target == 0: profit += (odds_2 - 1)
                    else: profit -= 1
                    invested += 1
                    
        roi = profit / invested if invested > 0 else -1.0
        return roi

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    best_w = study.best_params['w']
    best_roi = study.best_value
    
    print("-" * 55)
    print(f"Best Weight (XGB): {best_w:.4f}")
    print(f"Best ROI:          {best_roi:.2%}")
    
    # Update Params
    params['ensemble_xgb_weight'] = best_w
    with open(f'{BASE_DIR}/params.json', 'w') as f:
        json.dump({'best_params': params}, f, indent=4)
    print("Updated params.json with best ROI weight.")

if __name__ == "__main__":
    optimize_weight_roi()
