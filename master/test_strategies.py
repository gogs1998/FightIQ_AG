import pandas as pd
import numpy as np
import joblib
import json
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from models import SiameseMatchupNet, prepare_siamese_data

def test_strategies_and_calibration():
    print("=== Strategy & Calibration Testing ===")
    
    # 1. Load Data
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Split: Train Calibration on 2024, Test on 2025
    mask_calib = (df['event_date'].dt.year == 2024)
    mask_test = (df['event_date'].dt.year == 2025) & (df['f_1_odds'] > 1.0) & (df['f_2_odds'] > 1.0)
    
    df_calib = df[mask_calib].copy()
    df_test = df[mask_test].copy()
    
    print(f"Calibration Set (2024): {len(df_calib)}")
    print(f"Test Set (2025): {len(df_test)}")
    
    # 2. Load Model Pipeline
    xgb_model = joblib.load('models/xgb_optimized.pkl')
    scaler = joblib.load('models/siamese_scaler.pkl')
    
    with open('features.json', 'r') as f:
        features = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('models/siamese_optimized.pth', map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    # 3. Get Raw Probabilities
    def get_probs(dframe):
        X = dframe[[c for c in features if c in dframe.columns]].fillna(0)
        xgb_p = xgb_model.predict_proba(X)[:, 1]
        
        f1, f2, _, _ = prepare_siamese_data(X, features)
        f1 = scaler.transform(f1)
        f2 = scaler.transform(f2)
        t1 = torch.FloatTensor(f1).to(device)
        t2 = torch.FloatTensor(f2).to(device)
        with torch.no_grad():
            siam_p = siamese_model(t1, t2).cpu().numpy()
            
        return 0.405 * xgb_p + (1 - 0.405) * siam_p
        
    probs_calib = get_probs(df_calib)
    y_calib = df_calib['target'].values
    
    probs_test = get_probs(df_test)
    y_test = df_test['target'].values
    
    # 4. Calibrate (Platt Scaling)
    print("Training Platt Scaler (Logistic Regression) on 2024 data...")
    lr = LogisticRegression(C=1.0, solver='lbfgs')
    # Reshape for sklearn
    lr.fit(probs_calib.reshape(-1, 1), y_calib)
    
    probs_test_calibrated = lr.predict_proba(probs_test.reshape(-1, 1))[:, 1]
    
    # 5. Define Strategies
    strategies = [
        {"name": "Raw Model (Kelly 1/8)", "probs": probs_test, "cap_odds": 99.0, "kelly": 0.125},
        {"name": "Calibrated (Kelly 1/8)", "probs": probs_test_calibrated, "cap_odds": 99.0, "kelly": 0.125},
        {"name": "Raw + Max Odds 3.0", "probs": probs_test, "cap_odds": 3.0, "kelly": 0.125},
        {"name": "Calibrated + Max Odds 3.0", "probs": probs_test_calibrated, "cap_odds": 3.0, "kelly": 0.125},
        {"name": "Conservative (Kelly 1/16 + Cap 3.0)", "probs": probs_test_calibrated, "cap_odds": 3.0, "kelly": 0.0625},
    ]
    
    print(f"\n{'Strategy':<35} | {'ROI':<8} | {'Return':<8} | {'Min Bankroll':<12} | {'Final Bankroll':<12}")
    print("-" * 90)
    
    for strat in strategies:
        bankroll = 1000.0
        min_bankroll = 1000.0
        wagered = 0.0
        profit = 0.0
        
        p_arr = strat['probs']
        
        for i, prob in enumerate(p_arr):
            row = df_test.iloc[i]
            target = row['target']
            
            if prob > 0.5:
                odds = row['f_1_odds']
                my_prob = prob
                won = (target == 1)
            else:
                odds = row['f_2_odds']
                my_prob = 1 - prob
                won = (target == 0)
                
            # Odds Cap Filter
            if odds > strat['cap_odds']:
                continue
                
            # Kelly
            b = odds - 1.0
            q = 1.0 - my_prob
            f_star = (b * my_prob - q) / b
            
            if f_star > 0:
                stake = bankroll * f_star * strat['kelly']
                if stake > bankroll * 0.20: stake = bankroll * 0.20
                
                wagered += stake
                
                if won:
                    change = stake * (odds - 1)
                else:
                    change = -stake
                    
                bankroll += change
                profit += change
                if bankroll < min_bankroll: min_bankroll = bankroll
                
        roi = (profit / wagered * 100) if wagered > 0 else 0.0
        ret = ((bankroll - 1000) / 1000) * 100
        
        print(f"{strat['name']:<35} | {roi:>6.2f}% | {ret:>6.1f}% | ${min_bankroll:>10.2f} | ${bankroll:>10.2f}")

if __name__ == "__main__":
    test_strategies_and_calibration()
