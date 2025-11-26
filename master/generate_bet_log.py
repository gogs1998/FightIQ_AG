import pandas as pd
import numpy as np
import joblib
import json
import torch
from models import SiameseMatchupNet, prepare_siamese_data

def generate_2025_bet_log():
    print("Generating detailed 2025 bet log (Kelly 1/8)...")
    
    # 1. Load Data
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Filter for 2025 with valid odds
    df_2025 = df[(df['event_date'].dt.year == 2025) & 
                 (df['f_1_odds'] > 1.0) & 
                 (df['f_2_odds'] > 1.0)].copy()
    df_2025 = df_2025.sort_values('event_date')
    
    print(f"Found {len(df_2025)} betting opportunities in 2025.")
    
    # 2. Load Model Pipeline (With Odds)
    xgb_model = joblib.load('models/xgb_optimized.pkl')
    scaler = joblib.load('models/siamese_scaler.pkl')
    
    with open('features.json', 'r') as f:
        features = json.load(f)
        
    with open('models/siamese_cols.json', 'r') as f:
        siamese_cols = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Siamese
    state_dict = torch.load('models/siamese_optimized.pth', map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    # 3. Get Predictions
    X = df_2025[[c for c in features if c in df_2025.columns]].fillna(0)
    
    # XGB
    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    
    # Siamese
    f1_data, f2_data, _, _ = prepare_siamese_data(X, features)
    f1_data = scaler.transform(f1_data)
    f2_data = scaler.transform(f2_data)
    
    t_f1 = torch.FloatTensor(f1_data).to(device)
    t_f2 = torch.FloatTensor(f2_data).to(device)
    
    with torch.no_grad():
        siamese_probs = siamese_model(t_f1, t_f2).cpu().numpy()
        
    # Ensemble
    w = 0.405
    probs = w * xgb_probs + (1 - w) * siamese_probs
    
    # 4. Simulate Betting
    bankroll = 1000.0
    fraction = 0.125 # 1/8 Kelly
    bet_log = []
    
    for i, prob in enumerate(probs):
        row = df_2025.iloc[i]
        target = row['target']
        
        # Determine Bet
        if prob > 0.5:
            pred_fighter = row['f_1_name']
            opponent = row['f_2_name']
            odds = row['f_1_odds']
            my_prob = prob
            won_bet = (target == 1)
        else:
            pred_fighter = row['f_2_name']
            opponent = row['f_1_name']
            odds = row['f_2_odds']
            my_prob = 1 - prob
            won_bet = (target == 0)
            
        # Kelly Calc
        b = odds - 1.0
        q = 1.0 - my_prob
        f_star = (b * my_prob - q) / b
        
        if f_star > 0:
            stake = bankroll * f_star * fraction
            # Cap at 20%
            if stake > bankroll * 0.20: stake = bankroll * 0.20
            
            # Execute Bet
            implied_prob = 1 / odds
            edge = my_prob - implied_prob
            
            if won_bet:
                profit = stake * (odds - 1)
                result = "WIN"
            else:
                profit = -stake
                result = "LOSS"
                
            bankroll += profit
            
            bet_log.append({
                "Date": row['event_date'].strftime('%Y-%m-%d'),
                "Fighter": pred_fighter,
                "Opponent": opponent,
                "Odds": f"{odds:.2f}",
                "Model_Prob": f"{my_prob:.1%}",
                "Implied_Prob": f"{implied_prob:.1%}",
                "Edge": f"{edge:.1%}",
                "Stake": f"${stake:.2f}",
                "Result": result,
                "Profit": f"${profit:.2f}",
                "Bankroll": f"${bankroll:.2f}"
            })
            
    # Save to CSV
    log_df = pd.DataFrame(bet_log)
    log_df.to_csv('bet_log_2025_kelly_eighth.csv', index=False)
    print(f"Saved {len(bet_log)} bets to bet_log_2025_kelly_eighth.csv")
    print(f"Final Bankroll: ${bankroll:.2f}")

if __name__ == "__main__":
    generate_2025_bet_log()
