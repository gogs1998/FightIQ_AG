import pandas as pd
import numpy as np
import joblib
import json
import torch
from sklearn.isotonic import IsotonicRegression
from models import SiameseMatchupNet, prepare_siamese_data

def generate_final_bet_log():
    print("=== Generating Final Optimized Bet Log (Isotonic Calibration) ===")
    
    # 1. Load Data
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Calibration Set: 2023-2024
    mask_calib = (df['event_date'].dt.year.isin([2023, 2024]))
    # Test Set: 2025 (Valid Odds Only)
    mask_test = (df['event_date'].dt.year == 2025) & (df['f_1_odds'] > 1.0) & (df['f_2_odds'] > 1.0)
    
    df_calib = df[mask_calib].copy()
    df_test = df[mask_test].copy()
    
    y_calib = df_calib['target'].values
    
    print(f"Training Calibrators on {len(df_calib)} fights (2023-24)...")
    
    # 2. Load Models
    xgb_model = joblib.load('models/xgb_optimized.pkl')
    scaler = joblib.load('models/siamese_scaler.pkl')
    with open('features.json', 'r') as f: features = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('models/siamese_optimized.pth', map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    # 3. Get Raw Probabilities
    def get_component_probs(dframe):
        X = dframe[[c for c in features if c in dframe.columns]].fillna(0)
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        
        f1, f2, _, _ = prepare_siamese_data(X, features)
        f1 = scaler.transform(f1)
        f2 = scaler.transform(f2)
        t1 = torch.FloatTensor(f1).to(device)
        t2 = torch.FloatTensor(f2).to(device)
        with torch.no_grad():
            p_siam = siamese_model(t1, t2).cpu().numpy()
            
        return p_xgb, p_siam

    xgb_calib, siam_calib = get_component_probs(df_calib)
    xgb_test, siam_test = get_component_probs(df_test)
    
    # 4. Train Isotonic Calibrators
    iso_xgb = IsotonicRegression(out_of_bounds='clip')
    iso_xgb.fit(xgb_calib, y_calib)
    
    iso_siam = IsotonicRegression(out_of_bounds='clip')
    iso_siam.fit(siam_calib, y_calib)
    
    # 5. Apply Calibration
    p_xgb_iso = iso_xgb.predict(xgb_test)
    p_siam_iso = iso_siam.predict(siam_test)
    
    # Ensemble (Fixed Weight)
    w = 0.405
    final_probs = w * p_xgb_iso + (1 - w) * p_siam_iso
    
    # 6. Simulate Betting (Golden Rule Strategy)
    # Strategy: Min Edge 10%, Max Odds 3.0, Kelly 1/16, Conf > 60%
    
    bankroll = 1000.0
    bet_log = []
    
    for i, prob in enumerate(final_probs):
        row = df_test.iloc[i]
        target = row['target']
        
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
            
        # --- FILTERS ---
        if my_prob < 0.60: continue # Min Conf 60%
        if odds > 3.0: continue     # Max Odds 3.0
        
        implied = 1.0 / odds
        edge = my_prob - implied
        if edge < 0.10: continue    # Min Edge 10%
        
        # --- KELLY ---
        b = odds - 1.0
        q = 1.0 - my_prob
        f_star = (b * my_prob - q) / b
        
        if f_star > 0:
            stake = bankroll * f_star * 0.0625 # 1/16 Kelly
            if stake > bankroll * 0.20: stake = bankroll * 0.20
            
            if won_bet:
                profit = stake * (odds - 1)
                res_str = "WIN"
            else:
                profit = -stake
                res_str = "LOSS"
                
            bankroll += profit
            
            bet_log.append({
                "Date": row['event_date'].strftime('%Y-%m-%d'),
                "Fighter": pred_fighter,
                "Opponent": opponent,
                "Odds": f"{odds:.2f}",
                "Model_Prob": f"{my_prob:.1%}",
                "Edge": f"{edge:.1%}",
                "Stake": f"${stake:.2f}",
                "Result": res_str,
                "Profit": f"${profit:.2f}",
                "Bankroll": f"${bankroll:.2f}"
            })
            
    # Save Log
    log_df = pd.DataFrame(bet_log)
    log_df.to_csv('final_optimized_bets_2025.csv', index=False)
    
    print(f"\n=== Final Results (2025) ===")
    print(f"Total Bets: {len(bet_log)}")
    print(f"Final Bankroll: ${bankroll:.2f}")
    print(f"Total Profit: ${bankroll - 1000:.2f}")
    print(f"ROI: {(bankroll - 1000) / log_df['Stake'].str.replace('$','').astype(float).sum() * 100:.2f}%")
    print("Saved to final_optimized_bets_2025.csv")

if __name__ == "__main__":
    generate_final_bet_log()
