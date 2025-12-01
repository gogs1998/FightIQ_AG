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

from models import SiameseMatchupNet, prepare_siamese_data
from models.sequence_model import prepare_sequences
from models.opponent_adjustment import apply_opponent_adjustment

def generate_log():
    print("=== Generating Betting Log from Saved Models ===")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('data/training_data_enhanced.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Filter to 2024-2025 Holdout
    mask_test = df['event_date'] >= '2024-01-01'
    df_test = df[mask_test].copy()
    print(f"Test Set: {len(df_test)} fights (2024-2025)")
    
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
            
    # 3. Apply Opponent Adjustment
    adj_candidates = ['slpm_15_f_1', 'slpm_15_f_2', 'td_avg_15_f_1', 'td_avg_15_f_2', 'sub_avg_15_f_1', 'sub_avg_15_f_2', 'sapm_15_f_1', 'sapm_15_f_2']
    adj_cols = [c for c in adj_candidates if c in df.columns]
    if adj_cols and 'dynamic_elo_f1' in df.columns:
        df = apply_opponent_adjustment(df, adj_cols, elo_col='dynamic_elo')
        # Add adjusted cols to features
        for c in adj_cols:
            adj_name = f"{c}_adj"
            if adj_name not in features: features.append(adj_name)
            
    # 4. Prepare Data
    X_test = df.loc[mask_test, [c for c in features if c in df.columns]].fillna(0)
    
    # 5. Load Models
    print("Loading models...")
    xgb_model = joblib.load('models/xgb_master3.pkl')
    
    # Siamese
    seq_f1, seq_f2, seq_dim = prepare_sequences(df, features)
    seq_f1_test = seq_f1[mask_test]
    seq_f2_test = seq_f2[mask_test]
    
    f1_test, f2_test, input_dim, _ = prepare_siamese_data(X_test, features)
    
    # Scale (Fit on Training Data)
    mask_train = df['event_date'] < '2024-01-01'
    X_train = df.loc[mask_train, [c for c in features if c in df.columns]].fillna(0)
    f1_train, f2_train, _, _ = prepare_siamese_data(X_train, features)
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate([f1_train, f2_train], axis=0))
    f1_test = scaler.transform(f1_test)
    f2_test = scaler.transform(f2_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, seq_input_dim=seq_dim, hidden_dim=params.get('siamese_hidden_dim', 64)).to(device)
    siamese_model.load_state_dict(torch.load('models/siamese_master3.pth'))
    siamese_model.eval()
    
    # 6. Predict
    print("Predicting...")
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    with torch.no_grad():
        t_f1 = torch.FloatTensor(f1_test).to(device)
        t_f2 = torch.FloatTensor(f2_test).to(device)
        t_s1 = torch.FloatTensor(seq_f1_test).to(device)
        t_s2 = torch.FloatTensor(seq_f2_test).to(device)
        siamese_probs = siamese_model(t_f1, t_f2, t_s1, t_s2).cpu().numpy()
        
    w = params.get('ensemble_xgb_weight', 0.5)
    ens_probs = w * xgb_probs + (1 - w) * siamese_probs
    
    # 7. Generate Log
    print("Generating Betting Log...")
    df_test['prob_f1'] = ens_probs
    df_test['implied_f1'] = 1 / df_test['f_1_odds']
    df_test['implied_f2'] = 1 / df_test['f_2_odds']
    df_test['edge_f1'] = df_test['prob_f1'] - df_test['implied_f1']
    df_test['edge_f2'] = (1 - df_test['prob_f1']) - df_test['implied_f2']
    
    history = []
    min_edge = 0.05
    bankroll = 1000.0
    
    for idx, row in df_test.iterrows():
        bet_on = None
        odds = 0
        edge = 0
        
        if row['edge_f1'] > min_edge:
            bet_on = 'f1'
            odds = row['f_1_odds']
            edge = row['edge_f1']
        elif row['edge_f2'] > min_edge:
            bet_on = 'f2'
            odds = row['f_2_odds']
            edge = row['edge_f2']
            
        if bet_on:
            # Kelly Criterion
            # f = (bp - q) / b
            b = odds - 1
            p = row['prob_f1'] if bet_on == 'f1' else 1 - row['prob_f1']
            q = 1 - p
            
            f = (b * p - q) / b
            
            # Fractional Kelly (Safety)
            kelly_multiplier = 0.25 
            f = f * kelly_multiplier
            
            # Calculate Wager
            wager = bankroll * f
            if wager < 0: wager = 0
            
            # Safety Cap (Max 20% of bankroll on one fight)
            if wager > bankroll * 0.20: wager = bankroll * 0.20
            won = False
            if bet_on == 'f1' and row['target'] == 1: won = True
            if bet_on == 'f2' and row['target'] == 0: won = True
            
            if won:
                bankroll += wager * (odds - 1)
                res = 'WIN'
                profit = wager * (odds - 1)
            else:
                bankroll -= wager
                res = 'LOSS'
                profit = -wager
            
            history.append({
                'Date': row['event_date'],
                'Fighter1': row['f_1_name'],
                'Fighter2': row['f_2_name'],
                'Bet_On': row['f_1_name'] if bet_on == 'f1' else row['f_2_name'],
                'Odds': odds,
                'Edge': edge,
                'Prob': row['prob_f1'] if bet_on == 'f1' else 1 - row['prob_f1'],
                'Result': res,
                'Profit': profit,
                'Bankroll': bankroll
            })
            
    if history:
        log_df = pd.DataFrame(history)
        log_df.to_csv('betting_log.csv', index=False)
        print(f"Saved {len(log_df)} bets to betting_log.csv")
        print(f"Final Bankroll: ${bankroll:.2f}")
    else:
        print("No bets found.")

if __name__ == "__main__":
    generate_log()
