import requests
import pandas as pd
import numpy as np
import joblib
import json
import torch
import difflib
from models import SiameseMatchupNet, prepare_siamese_data

# API Config
API_KEY = '74b4ca301791b4b4c6ebe95897ac8673'
SPORT = 'mma_mixed_martial_arts'
REGIONS = 'uk'
MARKETS = 'h2h'
ODDS_FORMAT = 'decimal'

def clean_name(name):
    return name.strip().lower()

def find_fighter_in_db(name, db_names):
    if name in db_names: return name
    matches = difflib.get_close_matches(name, db_names, n=1, cutoff=0.7)
    if matches: return matches[0]
    return None

def get_upcoming_odds():
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}'
    try:
        return requests.get(url).json()
    except:
        return []

def construct_features(f1_name, f2_name, f1_odds, f2_odds, db, feature_names):
    stats_1 = db.get(f1_name)
    stats_2 = db.get(f2_name)
    if not stats_1 or not stats_2: return None
    row = {}
    for k, v in stats_1.items(): row[f'f_1_{k}'] = v
    for k, v in stats_2.items(): row[f'f_2_{k}'] = v
    row['f_1_odds'] = f1_odds
    row['f_2_odds'] = f2_odds
    row['diff_odds'] = f1_odds - f2_odds
    for feat in feature_names:
        if feat in row: continue
        val = 0.0
        if feat.startswith('diff_'):
            base = feat[5:]
            v1 = stats_1.get(base)
            v2 = stats_2.get(base)
            if v1 is not None and v2 is not None: val = v1 - v2
        elif feat.startswith('ratio_'):
            base = feat[6:]
            v1 = stats_1.get(base)
            v2 = stats_2.get(base)
            if v1 is not None and v2 is not None and v2 != 0: val = v1 / v2
        row[feat] = val
    for feat in feature_names:
        if feat not in row: row[feat] = 0.0
    return pd.DataFrame([row])

def analyze_taira():
    print("=== Analyzing Tatsuro Taira Matchup ===")
    
    # Load
    xgb_model = joblib.load('models/xgb_optimized.pkl')
    iso_xgb = joblib.load('models/iso_xgb.pkl')
    iso_siam = joblib.load('models/iso_siam.pkl')
    scaler = joblib.load('models/siamese_scaler.pkl')
    with open('features.json', 'r') as f: features = json.load(f)
    with open('fighter_db_production.json', 'r') as f: fighter_db = json.load(f)
    with open('models/siamese_cols.json', 'r') as f: siamese_cols = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('models/siamese_optimized.pth', map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    # Find Matchup
    events = get_upcoming_odds()
    target_match = None
    
    for event in events:
        for book in event['bookmakers']:
            for market in book['markets']:
                if market['key'] == 'h2h':
                    outcomes = market['outcomes']
                    names = [o['name'] for o in outcomes]
                    if "Tatsuro Taira" in names:
                        target_match = outcomes
                        break
            if target_match: break
        if target_match: break
        
    if not target_match:
        print("Could not find Tatsuro Taira in upcoming odds.")
        return

    f1_api = target_match[0]['name']
    f2_api = target_match[1]['name']
    o1 = target_match[0]['price']
    o2 = target_match[1]['price']
    
    print(f"Matchup Found: {f1_api} ({o1}) vs {f2_api} ({o2})")
    
    f1_db = find_fighter_in_db(f1_api, list(fighter_db.keys()))
    f2_db = find_fighter_in_db(f2_api, list(fighter_db.keys()))
    
    # Construct
    X = construct_features(f1_db, f2_db, o1, o2, fighter_db, features)
    
    # Predict
    X_xgb = X[[c for c in features if c in X.columns]]
    p_xgb_raw = xgb_model.predict_proba(X_xgb)[:, 1][0]
    
    f1_vec = []
    f2_vec = []
    for col in siamese_cols:
        val1 = 0.0
        if col in X.columns: val1 = X[col].values[0]
        f1_vec.append(val1)
        col2 = None
        if col.startswith('f_1_'): col2 = col.replace('f_1_', 'f_2_')
        elif '_f_1' in col: col2 = col.replace('_f_1', '_f_2')
        val2 = 0.0
        if col2 and col2 in X.columns: val2 = X[col2].values[0]
        f2_vec.append(val2)
        
    f1_feat = scaler.transform(np.array([f1_vec]))
    f2_feat = scaler.transform(np.array([f2_vec]))
    t1 = torch.FloatTensor(f1_feat).to(device)
    t2 = torch.FloatTensor(f2_feat).to(device)
    with torch.no_grad():
        p_siam_raw = siamese_model(t1, t2).cpu().numpy()
        if np.ndim(p_siam_raw) == 0: p_siam_raw = float(p_siam_raw)
        else: p_siam_raw = p_siam_raw[0]
        
    p_xgb_iso = iso_xgb.predict([p_xgb_raw])[0]
    p_siam_iso = iso_siam.predict([p_siam_raw])[0]
    final_prob = 0.405 * p_xgb_iso + (1 - 0.405) * p_siam_iso
    
    print(f"\n=== Model Analysis ===")
    print(f"XGBoost Raw: {p_xgb_raw:.1%} -> Calibrated: {p_xgb_iso:.1%}")
    print(f"Siamese Raw: {p_siam_raw:.1%} -> Calibrated: {p_siam_iso:.1%}")
    print(f"Final Probability: {final_prob:.1%}")
    
    # Feature Comparison
    s1 = fighter_db[f1_db]
    s2 = fighter_db[f2_db]
    
    print(f"\n=== Key Stats Comparison ===")
    print(f"{'Stat':<20} | {f1_db:<20} | {f2_db:<20} | {'Diff'}")
    print("-" * 70)
    
    keys = ['elo', 'age', 'reach_cm', 'slpm', 'sapm', 'td_avg', 'sub_avg']
    # Map to DB keys if different
    # DB keys are usually: elo, age, reach_cm, fighter_SlpM, fighter_SApM, td_avg_r1_15? No, need to check DB keys.
    # Let's just print what we have in s1/s2
    
    # Common keys
    common_keys = ['elo', 'age', 'reach_cm', 'fighter_SlpM', 'fighter_SApM', 'fighter_Str_Acc', 'fighter_Td_Avg', 'fighter_Sub_Avg']
    
    for k in common_keys:
        v1 = s1.get(k, 0)
        v2 = s2.get(k, 0)
        diff = v1 - v2 if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else "N/A"
        print(f"{k:<20} | {str(v1):<20} | {str(v2):<20} | {diff}")

if __name__ == "__main__":
    analyze_taira()
