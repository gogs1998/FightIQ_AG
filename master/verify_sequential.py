import pandas as pd
import numpy as np
import joblib
import json
import torch
from sklearn.isotonic import IsotonicRegression
from models import SiameseMatchupNet, prepare_siamese_data

def verify_sequential_betting():
    print("=== Verifying Sequential Betting (2024-2025) ===")
    
    # 1. Load Data
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date') # CRITICAL: Ensure chronological order
    
    # Calibration: 2023
    mask_calib = (df['event_date'].dt.year == 2023)
    # Test: 2024-2025
    mask_test = (df['event_date'].dt.year.isin([2024, 2025])) & (df['f_1_odds'] > 1.0) & (df['f_2_odds'] > 1.0)
    
    df_calib = df[mask_calib].copy()
    df_test = df[mask_test].copy()
    y_calib = df_calib['target'].values
    
    print(f"Calibration Set (2023): {len(df_calib)}")
    print(f"Test Set (2024-2025): {len(df_test)}")
    
    # 2. Load Pipeline
    xgb_model = joblib.load('models/xgb_optimized.pkl')
    scaler = joblib.load('models/siamese_scaler.pkl')
    with open('features.json', 'r') as f: features = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('models/siamese_optimized.pth', map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    # 3. Get Component Probs
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
    
    # Ensemble
    w = 0.405
    final_probs = w * p_xgb_iso + (1 - w) * p_siam_iso
    
    # 6. Sequential Simulation
    # Strategy: Max Odds 5.0, Kelly 1/4, Conf > 60%, Edge > 0%
    
    bankroll = 1000.0
    bet_log = []
    
    # Group by Date to simulate "Daily" betting
    # We place all bets for a specific date using the STARTING bankroll for that date
    # Then update bankroll after all events on that date are resolved
    
    df_test['prob'] = final_probs
    dates = df_test['event_date'].unique()
    
    print(f"Simulating {len(dates)} event dates sequentially...")
    
    for date in dates:
        daily_df = df_test[df_test['event_date'] == date]
        daily_bets = []
        daily_wagered = 0.0
        
        # 1. Place Bets (using current bankroll)
        for idx, row in daily_df.iterrows():
            prob = row['prob']
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
                
            # Filters
            if my_prob < 0.60: continue
            if odds > 5.0: continue
            
            implied = 1.0 / odds
            edge = my_prob - implied
            if edge < 0.0: continue
            
            # Kelly
            b = odds - 1.0
            q = 1.0 - my_prob
            f_star = (b * my_prob - q) / b
            
            if f_star > 0:
                # Use current bankroll for sizing
                stake = bankroll * f_star * 0.25 
                if stake > bankroll * 0.20: stake = bankroll * 0.20
                
                daily_bets.append({
                    "Date": str(date).split('T')[0],
                    "Fighter": pred_fighter,
                    "Opponent": opponent,
                    "Odds": odds,
                    "Prob": my_prob,
                    "Stake": stake,
                    "Won": won_bet,
                    "Profit": stake * (odds - 1) if won_bet else -stake
                })
                daily_wagered += stake
        
        # 2. Resolve Bets (Update bankroll)
        # Important: If we bet more than we have (impossible in reality), we need to scale down
        # But since we update sequentially per day, this is fine as long as daily_wagered < bankroll
        # If daily_wagered > bankroll, we would need to scale down all bets proportionally
        
        if daily_wagered > bankroll:
            scale_factor = bankroll / daily_wagered
            for bet in daily_bets:
                bet['Stake'] *= scale_factor
                bet['Profit'] *= scale_factor
                
        for bet in daily_bets:
            bankroll += bet['Profit']
            bet_log.append({
                "Date": bet['Date'],
                "Fighter": bet['Fighter'],
                "Opponent": bet['Opponent'],
                "Odds": f"{bet['Odds']:.2f}",
                "Prob": f"{bet['Prob']:.1%}",
                "Stake": f"${bet['Stake']:.2f}",
                "Result": "WIN" if bet['Won'] else "LOSS",
                "Profit": f"${bet['Profit']:.2f}",
                "Bankroll": f"${bankroll:.2f}"
            })
            
    # Save Log
    log_df = pd.DataFrame(bet_log)
    log_df.to_csv('final_sequential_bets.csv', index=False)
    
    print(f"\n=== Final Sequential Results (2024-2025) ===")
    print(f"Total Bets: {len(bet_log)}")
    print(f"Final Bankroll: ${bankroll:.2f}")
    print(f"Total Profit: ${bankroll - 1000:.2f}")
    
    # Calculate ROI
    total_stake = log_df['Stake'].str.replace('$','').astype(float).sum()
    roi = (bankroll - 1000) / total_stake * 100
    print(f"ROI: {roi:.2f}%")

if __name__ == "__main__":
    verify_sequential_betting()
