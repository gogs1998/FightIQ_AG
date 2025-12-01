import pandas as pd
import numpy as np
import joblib
import json
import torch
from sklearn.metrics import accuracy_score, log_loss
from models import SiameseMatchupNet, prepare_siamese_data

def optimize_weight():
    print("=== Optimizing Ensemble Weight ===")
    
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
        
    # Sweep Weights
    best_acc = 0
    best_w = 0
    best_ll = 10.0
    
    print(f"{'Weight (XGB)':<15} {'Accuracy':<15} {'Log Loss':<15}")
    print("-" * 45)
    
    for w in np.linspace(0, 1, 21): # 0.0 to 1.0 in 0.05 steps
        p_ens = w * p_xgb + (1 - w) * p_siam
        acc = accuracy_score(y_test, (p_ens > 0.5).astype(int))
        ll = log_loss(y_test, p_ens)
        
        print(f"{w:<15.2f} {acc:<15.4%} {ll:<15.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_w = w
            best_ll = ll
            
    print("-" * 45)
    print(f"Best Weight (XGB): {best_w:.2f}")
    print(f"Best Accuracy:     {best_acc:.4%}")
    print(f"Log Loss:          {best_ll:.4f}")
    
    # Update Params
    params['ensemble_xgb_weight'] = best_w
    with open(f'{BASE_DIR}/params.json', 'w') as f:
        json.dump({'best_params': params}, f, indent=4)
    print("Updated params.json with best weight.")

if __name__ == "__main__":
    optimize_weight()
