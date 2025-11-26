import pandas as pd
import numpy as np
import joblib
import json
import torch
from models import SiameseMatchupNet, prepare_siamese_data

def debug_roi_discrepancy():
    print("=== Debugging ROI Discrepancy ===")
    
    # 1. Load Data
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Filter for 2025 with valid odds
    df_2025 = df[(df['event_date'].dt.year == 2025) & 
                 (df['f_1_odds'] > 1.0) & 
                 (df['f_2_odds'] > 1.0)].copy()
    df_2025 = df_2025.sort_values('event_date')
    
    print(f"Dataset Size: {len(df_2025)}")
    
    # 2. Load Model Pipeline
    xgb_model = joblib.load('models/xgb_optimized.pkl')
    scaler = joblib.load('models/siamese_scaler.pkl')
    
    with open('features.json', 'r') as f:
        features = json.load(f)
        
    with open('models/siamese_cols.json', 'r') as f:
        siamese_cols = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('models/siamese_optimized.pth', map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    # 3. Get Predictions
    X = df_2025[[c for c in features if c in df_2025.columns]].fillna(0)
    
    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    
    f1_data, f2_data, _, _ = prepare_siamese_data(X, features)
    f1_data = scaler.transform(f1_data)
    f2_data = scaler.transform(f2_data)
    
    t_f1 = torch.FloatTensor(f1_data).to(device)
    t_f2 = torch.FloatTensor(f2_data).to(device)
    
    with torch.no_grad():
        siamese_probs = siamese_model(t_f1, t_f2).cpu().numpy()
        
    w = 0.405
    probs = w * xgb_probs + (1 - w) * siamese_probs
    
    # 4. Run Simulation 1 (The Logic from verify_roi_2025.py)
    print("\n--- Sim 1: Logic from verify_roi_2025.py ---")
    bankroll = 1000.0
    fraction = 0.125
    
    for i, prob in enumerate(probs):
        row = df_2025.iloc[i]
        
        # Logic: Predict based on prob > 0.5
        pred = 1 if prob > 0.5 else 0
        
        if pred == 1:
            odds = row['f_1_odds']
            target = row['target'] # 1 means f1 won
            won = (target == 1)
            my_prob = prob
        else:
            odds = row['f_2_odds']
            target = row['target'] # 0 means f2 won
            won = (target == 0)
            my_prob = 1 - prob
            
        # Kelly
        b = odds - 1.0
        q = 1.0 - my_prob
        f_star = (b * my_prob - q) / b
        
        if f_star > 0:
            stake = bankroll * f_star * fraction
            if stake > bankroll * 0.20: stake = bankroll * 0.20
            
            if won:
                bankroll += stake * (odds - 1)
            else:
                bankroll -= stake
                
    print(f"Final Bankroll (Sim 1): ${bankroll:.2f}")
    
    # 5. Run Simulation 2 (The Logic from generate_bet_log.py)
    print("\n--- Sim 2: Logic from generate_bet_log.py ---")
    # Check if there is any difference in logic...
    # Reading code... logic seems identical.
    # Let's check if the DATA is different.
    # verify_roi_2025.py used: split_date = '2025-01-01'
    # generate_bet_log.py used: df['event_date'].dt.year == 2025
    
    # Let's check the number of bets placed in Sim 1
    # ... (re-running loop to count)
    
    # Wait, I suspect the issue might be the MODEL state or DATA loading.
    # Did we retrain the model in between?
    # Or did we use a different feature set?
    
    # Let's print the first 5 predictions to compare with the CSV log
    print("\nFirst 5 Predictions:")
    for i in range(5):
        print(f"{df_2025.iloc[i]['f_1_name']} vs {df_2025.iloc[i]['f_2_name']}: Prob {probs[i]:.4f}")

if __name__ == "__main__":
    debug_roi_discrepancy()
