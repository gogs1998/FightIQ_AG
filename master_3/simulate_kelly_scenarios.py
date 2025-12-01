import pandas as pd
import numpy as np
import joblib
import json
import torch
from models import SiameseMatchupNet, prepare_siamese_data

# --- Configuration ---
BASE_DIR = 'd:/AntiGravity/FightIQ/master_2'
INITIAL_BANKROLL = 1000.0
MAX_WIN_PER_EVENT = 50000.0 # Realistic Bookmaker Limit

def load_test_predictions():
    print("Loading Data & Models...")
    df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    with open(f'{BASE_DIR}/features.json', 'r') as f: features = json.load(f)
    with open(f'{BASE_DIR}/params.json', 'r') as f: params = json.load(f)['best_params']
    
    # Filter Odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Split Test Set
    mask_test = df['event_date'] >= '2024-01-01'
    test_df = df[mask_test].copy()
    X_df = df[[c for c in features if c in df.columns]].fillna(0)
    X_test = X_df[mask_test]
    
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
    
    # Generate Predictions
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    with torch.no_grad():
        t1 = torch.FloatTensor(f1_test).to(device)
        t2 = torch.FloatTensor(f2_test).to(device)
        p_siam = siamese_model(t1, t2).cpu().numpy().flatten()
        
    w = params['ensemble_xgb_weight']
    p_ens = w * p_xgb + (1 - w) * p_siam
    
    test_df['prob'] = p_ens
    return test_df

def simulate_strategy(df, kelly_fraction, min_edge=0.013, min_conf=0.45, max_odds=5.8):
    bankroll = INITIAL_BANKROLL
    history = [bankroll]
    stakes = []
    
    for _, row in df.iterrows():
        prob = row['prob']
        target = row['target']
        odds_1 = row['f_1_odds']
        odds_2 = row['f_2_odds']
        
        if prob > 0.5:
            my_prob = prob
            odds = odds_1
            win = (target == 1)
        else:
            my_prob = 1 - prob
            odds = odds_2
            win = (target == 0)
            
        # Filters
        if odds > max_odds: continue
        if my_prob < min_conf: continue
        
        implied = 1 / odds
        edge = my_prob - implied
        if edge < min_edge: continue
        
        # Kelly Criterion
        b = odds - 1
        q = 1 - my_prob
        f = (b * my_prob - q) / b
        if f < 0: f = 0
        
        # Stake Calculation
        raw_stake = bankroll * f * kelly_fraction
        
        # Apply Limits
        # 1. Bankroll Limit (Safety)
        if raw_stake > bankroll * 0.20: raw_stake = bankroll * 0.20
        
        # 2. Max Win Limit ($50k)
        max_stake_for_limit = MAX_WIN_PER_EVENT / (odds - 1)
        stake = min(raw_stake, max_stake_for_limit)
        
        if stake < 5: stake = 0 # Minimum bet
        
        if stake > 0:
            stakes.append(stake)
            if win:
                bankroll += stake * (odds - 1)
            else:
                bankroll -= stake
                
        history.append(bankroll)
        if bankroll < 10: break # Ruin
        
    # Metrics
    hist_s = pd.Series(history)
    peak = hist_s.cummax()
    dd = (peak - hist_s) / peak
    max_dd = dd.max()
    profit = bankroll - INITIAL_BANKROLL
    roi = profit / sum(stakes) if stakes else 0
    
    return {
        "Kelly": kelly_fraction,
        "Final Bankroll": bankroll,
        "Profit": profit,
        "Max Drawdown": max_dd,
        "ROI (Yield)": roi,
        "Bets Placed": len(stakes),
        "Avg Stake": np.mean(stakes) if stakes else 0,
        "Max Stake": np.max(stakes) if stakes else 0
    }

def run_scenarios():
    df = load_test_predictions()
    
    fractions = [0.10, 0.125, 0.25, 0.33, 0.50, 0.75, 1.00]
    results = []
    
    print(f"\n=== Simulation Results (Max Win Limit: ${MAX_WIN_PER_EVENT:,.0f}) ===")
    
    for frac in fractions:
        res = simulate_strategy(df, frac)
        results.append(res)
        print(f"Kelly {frac}: Bank ${res['Final Bankroll']:,.0f}")
        
    with open(f'{BASE_DIR}/simulation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Saved results to simulation_results.json")

if __name__ == "__main__":
    run_scenarios()
