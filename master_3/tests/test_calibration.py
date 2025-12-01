import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import sys
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import SiameseMatchupNet, prepare_siamese_data
from models.sequence_model import prepare_sequences
from models.opponent_adjustment import apply_opponent_adjustment

def run_calibration_test():
    print("=== Master 3: Calibration Test (2024-2025) ===")
    
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists('data/training_data_enhanced.csv'):
        print("Error: data/training_data_enhanced.csv not found.")
        return
        
    df = pd.read_csv('data/training_data_enhanced.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 2. Load Config
    if os.path.exists('features_selected.json'):
        with open('features_selected.json', 'r') as f:
            features = json.load(f)
    else:
        with open('features_enhanced.json', 'r') as f:
            features = json.load(f)
            
    with open('params.json', 'r') as f:
        best = json.load(f)
        params = best['best_params']
    if os.path.exists('params_optimized.json'):
        with open('params_optimized.json', 'r') as f:
            params.update(json.load(f))
            
    # 3. Prepare Data (Same as train.py)
    # Apply Opponent Adjustment
    adj_candidates = ['slpm_15_f_1', 'slpm_15_f_2', 'td_avg_15_f_1', 'td_avg_15_f_2', 'sub_avg_15_f_1', 'sub_avg_15_f_2', 'sapm_15_f_1', 'sapm_15_f_2']
    adj_cols = [c for c in adj_candidates if c in df.columns]
    if adj_cols and 'dynamic_elo_f1' in df.columns:
        df = apply_opponent_adjustment(df, adj_cols, elo_col='dynamic_elo')
        for c in adj_cols:
            adj_name = f"{c}_adj"
            if adj_name not in features: features.append(adj_name)
            
    # Filter Odds
    if 'f_1_odds' in features and 'f_2_odds' in features:
        has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
        df = df[has_odds].copy()
        
    X_df = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    # Split
    split_date = '2024-01-01'
    mask_train = df['event_date'] < split_date
    mask_test = df['event_date'] >= split_date
    
    X_train = X_df[mask_train]
    X_test = X_df[mask_test]
    y_test = y[mask_test]
    
    print(f"Test Set Size: {len(X_test)}")
    
    # 4. Load Models & Predict
    print("Loading models...")
    xgb_model = joblib.load('models/xgb_master3.pkl')
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Siamese
    # Need to prep sequences and siamese data
    seq_f1, seq_f2, seq_dim = prepare_sequences(df, features)
    seq_f1_test = seq_f1[mask_test]
    seq_f2_test = seq_f2[mask_test]
    
    f1_train, f2_train, input_dim, _ = prepare_siamese_data(X_train, features)
    f1_test, f2_test, _, _ = prepare_siamese_data(X_test, features)
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate([f1_train, f2_train], axis=0))
    f1_test = scaler.transform(f1_test)
    f2_test = scaler.transform(f2_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, seq_input_dim=seq_dim, hidden_dim=params.get('siamese_hidden_dim', 64)).to(device)
    
    if os.path.exists('models/siamese_master3.pth'):
        siamese_model.load_state_dict(torch.load('models/siamese_master3.pth'))
    else:
        print("Error: Siamese weights not found.")
        return
        
    siamese_model.eval()
    with torch.no_grad():
        t_f1 = torch.FloatTensor(f1_test).to(device)
        t_f2 = torch.FloatTensor(f2_test).to(device)
        t_s1 = torch.FloatTensor(seq_f1_test).to(device)
        t_s2 = torch.FloatTensor(seq_f2_test).to(device)
        siamese_probs = siamese_model(t_f1, t_f2, t_s1, t_s2).cpu().numpy()
        
    # Ensemble
    w = params.get('ensemble_xgb_weight', 0.5)
    ens_probs = w * xgb_probs + (1 - w) * siamese_probs
    
    # 5. Calibration Analysis
    print("\n--- Calibration Results ---")
    
    brier = brier_score_loss(y_test, ens_probs)
    print(f"Brier Score: {brier:.4f} (Lower is better)")
    
    prob_true, prob_pred = calibration_curve(y_test, ens_probs, n_bins=10)
    
    print("\nReliability Diagram (Text):")
    print(f"{'Mean Pred':<10} | {'Fraction Pos':<12} | {'Count':<5} | {'Error':<10}")
    print("-" * 45)
    
    ece = 0.0
    total_count = len(y_test)
    
    # We need counts per bin to calc ECE manually or use a library
    # Let's do it manually to be precise
    bins = np.linspace(0, 1, 11)
    binids = np.digitize(ens_probs, bins) - 1
    
    for i in range(10):
        mask = binids == i
        if np.sum(mask) > 0:
            mean_pred = np.mean(ens_probs[mask])
            fraction_pos = np.mean(y_test[mask])
            count = np.sum(mask)
            error = abs(mean_pred - fraction_pos)
            ece += (count / total_count) * error
            
            print(f"{mean_pred:.4f}     | {fraction_pos:.4f}       | {count:<5} | {error:.4f}")
            
    print("-" * 45)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    if ece < 0.05:
        print("STATUS: Excellent Calibration (< 0.05)")
    elif ece < 0.10:
        print("STATUS: Good Calibration (< 0.10)")
    else:
        print("STATUS: Poor Calibration (> 0.10). Consider Platt Scaling.")

if __name__ == "__main__":
    run_calibration_test()
