import pandas as pd
import numpy as np
import joblib
import json
import torch
from sklearn.isotonic import IsotonicRegression
from models import SiameseMatchupNet, prepare_siamese_data

def optimize_strategy_isotonic_2024():
    print("=== Strategy Optimization Grid Search (Isotonic Calibration - 2024) ===")
    
    # 1. Load Data
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Calibration: 2023 (Train calibration on previous year)
    mask_calib = (df['event_date'].dt.year == 2023)
    # Test: 2024
    mask_test = (df['event_date'].dt.year == 2024) & (df['f_1_odds'] > 1.0) & (df['f_2_odds'] > 1.0)
    
    df_calib = df[mask_calib].copy()
    df_test = df[mask_test].copy()
    y_calib = df_calib['target'].values
    
    print(f"Calibration Set (2023): {len(df_calib)}")
    print(f"Test Set (2024): {len(df_test)}")
    
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
    probs_test_iso = w * p_xgb_iso + (1 - w) * p_siam_iso
    
    # 6. Grid Search
    grid = []
    
    min_edges = [0.0, 0.02, 0.05, 0.10, 0.15]
    max_odds_list = [2.5, 3.0, 3.5, 4.0, 5.0, 99.0]
    kelly_fracs = [0.25, 0.125, 0.0625]
    conf_thresholds = [0.50, 0.55, 0.60, 0.65]
    
    print(f"\nTesting {len(min_edges)*len(max_odds_list)*len(kelly_fracs)*len(conf_thresholds)} combinations...")
    
    best_profit = -1000.0
    best_config = None
    
    for edge_min in min_edges:
        for max_odds in max_odds_list:
            for k_frac in kelly_fracs:
                for conf in conf_thresholds:
                    
                    bankroll = 1000.0
                    wagered = 0.0
                    profit = 0.0
                    bets = 0
                    
                    for i, prob in enumerate(probs_test_iso):
                        row = df_test.iloc[i]
                        target = row['target']
                        
                        if prob > 0.5:
                            my_prob = prob
                            odds = row['f_1_odds']
                            won = (target == 1)
                        else:
                            my_prob = 1 - prob
                            odds = row['f_2_odds']
                            won = (target == 0)
                            
                        if my_prob < conf: continue
                        if odds > max_odds: continue
                        
                        implied = 1.0 / odds
                        edge = my_prob - implied
                        if edge < edge_min: continue
                        
                        b = odds - 1.0
                        q = 1.0 - my_prob
                        f_star = (b * my_prob - q) / b
                        
                        if f_star > 0:
                            stake = bankroll * f_star * k_frac
                            if stake > bankroll * 0.20: stake = bankroll * 0.20
                            
                            wagered += stake
                            bets += 1
                            
                            if won:
                                change = stake * (odds - 1)
                            else:
                                change = -stake
                                
                            bankroll += change
                            profit += change
                            
                    roi = (profit / wagered * 100) if wagered > 0 else 0.0
                    
                    if bets > 10 and profit > best_profit:
                        best_profit = profit
                        best_config = {
                            "Edge": edge_min,
                            "MaxOdds": max_odds,
                            "Kelly": k_frac,
                            "Conf": conf,
                            "ROI": roi,
                            "Bets": bets,
                            "Profit": profit,
                            "Final": bankroll
                        }
                        
    print("\n=== Best Strategy Found (Isotonic - 2024) ===")
    print(f"Total Profit: ${best_config['Profit']:.2f}")
    print(f"Final Bankroll: ${best_config['Final']:.2f}")
    print(f"ROI: {best_config['ROI']:.2f}%")
    print(f"Bets Placed: {best_config['Bets']}")
    print("\nParameters:")
    print(f"  Min Edge: {best_config['Edge']:.1%}")
    print(f"  Max Odds: {best_config['MaxOdds']}")
    print(f"  Kelly Fraction: {best_config['Kelly']}")
    print(f"  Min Confidence: {best_config['Conf']:.0%}")

if __name__ == "__main__":
    optimize_strategy_isotonic_2024()
