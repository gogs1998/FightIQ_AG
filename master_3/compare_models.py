import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import sys
import torch
from sklearn.preprocessing import StandardScaler

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Master 3 Utils
from models import SiameseMatchupNet, prepare_siamese_data
from models.sequence_model import prepare_sequences
from models.opponent_adjustment import apply_opponent_adjustment

def compare_models():
    print("=== Model Comparison: Master 3 vs Master 2 (Baseline) ===")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('data/training_data_enhanced.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Filter to 2024-2025 Holdout
    mask_test = df['event_date'] >= '2024-01-01'
    df_test = df[mask_test].copy()
    print(f"Test Set: {len(df_test)} fights (2024-2025)")
    
    # ---------------------------------------------------------
    # MASTER 3 (The New Model)
    # ---------------------------------------------------------
    print("\n--- Generating Master 3 Predictions ---")
    # Load Config
    if os.path.exists('features_selected.json'):
        with open('features_selected.json', 'r') as f:
            features_m3 = json.load(f)
    else:
        with open('features_enhanced.json', 'r') as f:
            features_m3 = json.load(f)
            
    with open('params.json', 'r') as f:
        best = json.load(f)
        params_m3 = best['best_params']
    if os.path.exists('params_optimized.json'):
        with open('params_optimized.json', 'r') as f:
            params_m3.update(json.load(f))
            
    # Apply Opponent Adjustment (M3 only)
    adj_candidates = ['slpm_15_f_1', 'slpm_15_f_2', 'td_avg_15_f_1', 'td_avg_15_f_2', 'sub_avg_15_f_1', 'sub_avg_15_f_2', 'sapm_15_f_1', 'sapm_15_f_2']
    adj_cols = [c for c in adj_candidates if c in df.columns]
    if adj_cols and 'dynamic_elo_f1' in df.columns:
        df_m3 = apply_opponent_adjustment(df, adj_cols, elo_col='dynamic_elo')
        # Add adjusted cols to features if not already there (CRITICAL FIX)
        for c in adj_cols:
            adj_name = f"{c}_adj"
            if adj_name not in features_m3:
                features_m3.append(adj_name)
    else:
        df_m3 = df.copy()
        
    # Prepare M3 Data
    X_m3 = df_m3.loc[mask_test, [c for c in features_m3 if c in df_m3.columns]].fillna(0)
    
    # Load M3 Models
    xgb_m3 = joblib.load('models/xgb_master3.pkl')
    
    # Siamese M3
    seq_f1, seq_f2, seq_dim = prepare_sequences(df_m3, features_m3)
    seq_f1_test = seq_f1[mask_test]
    seq_f2_test = seq_f2[mask_test]
    
    f1_test, f2_test, input_dim, _ = prepare_siamese_data(X_m3, features_m3)
    
    # Scale M3 (Need to fit scaler on training data to be fair)
    # Quick hack: Fit on test data? No, that's leakage.
    # Fit on training data.
    mask_train = df['event_date'] < '2024-01-01'
    X_train_m3 = df_m3.loc[mask_train, [c for c in features_m3 if c in df_m3.columns]].fillna(0)
    f1_train, f2_train, _, _ = prepare_siamese_data(X_train_m3, features_m3)
    scaler_m3 = StandardScaler()
    scaler_m3.fit(np.concatenate([f1_train, f2_train], axis=0))
    
    f1_test_scaled = scaler_m3.transform(f1_test)
    f2_test_scaled = scaler_m3.transform(f2_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siam_m3 = SiameseMatchupNet(input_dim, seq_input_dim=seq_dim, hidden_dim=params_m3.get('siamese_hidden_dim', 64)).to(device)
    if os.path.exists('models/siamese_master3.pth'):
        siam_m3.load_state_dict(torch.load('models/siamese_master3.pth'))
    siam_m3.eval()
    
    # Predict M3
    xgb_probs_m3 = xgb_m3.predict_proba(X_m3)[:, 1]
    with torch.no_grad():
        t_f1 = torch.FloatTensor(f1_test_scaled).to(device)
        t_f2 = torch.FloatTensor(f2_test_scaled).to(device)
        t_s1 = torch.FloatTensor(seq_f1_test).to(device)
        t_s2 = torch.FloatTensor(seq_f2_test).to(device)
        siam_probs_m3 = siam_m3(t_f1, t_f2, t_s1, t_s2).cpu().numpy()
        
    w_m3 = params_m3.get('ensemble_xgb_weight', 0.5)
    probs_m3 = w_m3 * xgb_probs_m3 + (1 - w_m3) * siam_probs_m3
    
    # ---------------------------------------------------------
    # MASTER 2 (The Baseline)
    # ---------------------------------------------------------
    print("\n--- Generating Master 2 Predictions ---")
    m2_path = '../master_2'
    
    # Load M2 Features
    with open(f'{m2_path}/features.json', 'r') as f:
        features_m2 = json.load(f)
        
    # Prepare M2 Data (No Opponent Adjustment, No Sequences)
    # Note: Master 2 used 'training_data.csv', not 'enhanced'.
    # But 'enhanced' is a superset, so it should have the features.
    # We just need to select them.
    X_m2 = df.loc[mask_test, [c for c in features_m2 if c in df.columns]].fillna(0)
    
    # Load M2 Models
    xgb_m2 = joblib.load(f'{m2_path}/models/xgb_optimized.pkl')
    scaler_m2 = joblib.load(f'{m2_path}/models/siamese_scaler.pkl')
    
    # Siamese M2 (Different Architecture - No Sequences)
    # We need to instantiate the OLD Siamese class.
    # But we imported the NEW one.
    # Hack: The new one has `seq_input_dim`. If we pass 0, maybe it works?
    # No, the forward pass expects 4 args.
    # We need to define the old class or import it from master_2.
    # Let's import it dynamically.
    sys.path.append(os.path.abspath(m2_path))
    from models import SiameseMatchupNet as SiameseM2
    
    # Prepare M2 Siamese Data
    f1_test_m2, f2_test_m2, input_dim_m2, _ = prepare_siamese_data(X_m2, features_m2)
    f1_test_m2 = scaler_m2.transform(f1_test_m2)
    f2_test_m2 = scaler_m2.transform(f2_test_m2)
    
    # Load Params M2
    with open(f'{m2_path}/params.json', 'r') as f:
        params_m2 = json.load(f)['best_params']
        
    siam_m2 = SiameseM2(input_dim_m2, hidden_dim=params_m2['siamese_hidden_dim']).to(device)
    siam_m2.load_state_dict(torch.load(f'{m2_path}/models/siamese_optimized.pth'))
    siam_m2.eval()
    
    # Predict M2
    xgb_probs_m2 = xgb_m2.predict_proba(X_m2)[:, 1]
    with torch.no_grad():
        t_f1 = torch.FloatTensor(f1_test_m2).to(device)
        t_f2 = torch.FloatTensor(f2_test_m2).to(device)
        siam_probs_m2 = siam_m2(t_f1, t_f2).cpu().numpy()
        
    w_m2 = params_m2['ensemble_xgb_weight']
    probs_m2 = w_m2 * xgb_probs_m2 + (1 - w_m2) * siam_probs_m2
    
    # ---------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------
    print("\n--- Comparison Analysis ---")
    
    results = df_test[['event_date', 'f_1_name', 'f_2_name', 'target', 'f_1_odds', 'f_2_odds']].copy()
    results['Prob_M3'] = probs_m3
    results['Prob_M2'] = probs_m2
    
    results['Pred_M3'] = (probs_m3 > 0.5).astype(int)
    results['Pred_M2'] = (probs_m2 > 0.5).astype(int)
    
    # Disagreements
    results['Disagree'] = results['Pred_M3'] != results['Pred_M2']
    
    # Value Bets (Edge > 5%)
    results['Implied_F1'] = 1 / results['f_1_odds']
    results['Edge_M3'] = results['Prob_M3'] - results['Implied_F1']
    results['Edge_M2'] = results['Prob_M2'] - results['Implied_F1']
    
    results['Bet_M3'] = results['Edge_M3'] > 0.05
    results['Bet_M2'] = results['Edge_M2'] > 0.05
    
    # Filter to Disagreements
    disagreements = results[results['Disagree']].copy()
    print(f"Total Fights: {len(results)}")
    print(f"Disagreements: {len(disagreements)} ({len(disagreements)/len(results):.1%})")
    
    # Save Log
    cols = ['event_date', 'f_1_name', 'f_2_name', 'target', 'Prob_M3', 'Prob_M2', 'Bet_M3', 'Bet_M2']
    results.to_csv('model_comparison_full.csv', index=False)
    disagreements[cols].to_csv('model_disagreements.csv', index=False)
    
    print("\nTop 10 Disagreements (Where M3 was Right and M2 was Wrong):")
    m3_wins = disagreements[disagreements['Pred_M3'] == disagreements['target']]
    print(m3_wins[cols].head(10).to_string(index=False))
    
    print("\nTop 10 Disagreements (Where M2 was Right and M3 was Wrong):")
    m2_wins = disagreements[disagreements['Pred_M2'] == disagreements['target']]
    print(m2_wins[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    compare_models()
