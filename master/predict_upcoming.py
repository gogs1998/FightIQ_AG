import requests
import pandas as pd
import numpy as np
import joblib
import json
import torch
import difflib
from datetime import datetime
from models import SiameseMatchupNet, prepare_siamese_data

# API Config
API_KEY = '74b4ca301791b4b4c6ebe95897ac8673'
SPORT = 'mma_mixed_martial_arts'
REGIONS = 'uk' # Changed to UK
MARKETS = 'h2h'
ODDS_FORMAT = 'decimal'

def clean_name(name):
    return name.strip().lower()

def find_fighter_in_db(name, db_names):
    # 1. Exact match
    if name in db_names: return name
    
    # 2. Case insensitive
    name_lower = name.lower()
    for n in db_names:
        if n.lower() == name_lower: return n
        
    # 3. Fuzzy match
    matches = difflib.get_close_matches(name, db_names, n=1, cutoff=0.7)
    if matches: return matches[0]
    
    return None

def get_upcoming_odds():
    print("Fetching upcoming odds from API...")
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions={REGIONS}&markets={MARKETS}&oddsFormat={ODDS_FORMAT}'
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"API Error: {response.text}")
            return []
        return response.json()
    except Exception as e:
        print(f"Connection Error: {e}")
        return []

def construct_features(f1_name, f2_name, f1_odds, f2_odds, db, feature_names):
    stats_1 = db.get(f1_name)
    stats_2 = db.get(f2_name)
    
    if not stats_1 or not stats_2:
        return None
        
    row = {}
    
    # 1. Add ALL raw stats from DB to row (for Siamese)
    for k, v in stats_1.items():
        row[f'f_1_{k}'] = v
    for k, v in stats_2.items():
        row[f'f_2_{k}'] = v
        
    # 2. Add Odds
    row['f_1_odds'] = f1_odds
    row['f_2_odds'] = f2_odds
    row['diff_odds'] = f1_odds - f2_odds
    
    # 3. Add Diffs/Ratios (Explicitly for XGBoost features)
    for feat in feature_names:
        if feat in row: continue # Already added
        
        val = 0.0
        
        if feat.startswith('diff_'):
            base = feat[5:]
            v1 = stats_1.get(base)
            v2 = stats_2.get(base)
            if v1 is not None and v2 is not None:
                val = v1 - v2
        elif feat.startswith('ratio_'):
            base = feat[6:]
            v1 = stats_1.get(base)
            v2 = stats_2.get(base)
            if v1 is not None and v2 is not None and v2 != 0:
                val = v1 / v2
                
        row[feat] = val
        
    # Ensure all expected features are present (fill missing with 0)
    for feat in feature_names:
        if feat not in row:
            row[feat] = 0.0
            
    return pd.DataFrame([row])

def predict_upcoming():
    print("=== FightIQ: Upcoming Fight Predictor (UK Region) ===")
    
    # 1. Load Resources
    print("Loading models and database...")
    try:
        xgb_model = joblib.load('models/xgb_production.pkl')
        iso_xgb = joblib.load('models/iso_xgb_production.pkl')
        iso_siam = joblib.load('models/iso_siam_production.pkl')
        scaler = joblib.load('models/siamese_scaler_production.pkl')
        
        with open('features.json', 'r') as f: features = json.load(f)
        with open('fighter_db_production.json', 'r') as f: fighter_db = json.load(f)
        with open('models/siamese_cols.json', 'r') as f: siamese_cols = json.load(f)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load('models/siamese_production.pth', map_location=device)
        input_dim = state_dict['encoder.0.weight'].shape[1]
        siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
        siamese_model.load_state_dict(state_dict)
        siamese_model.eval()
        
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    # 2. Get Odds
    events = get_upcoming_odds()
    if not events:
        print("No upcoming events found.")
        return
        
    print(f"Found {len(events)} upcoming events.")
    
    db_names = list(fighter_db.keys())
    
    bets_to_make = []
    
    # UK Bookmakers to check
    uk_bookies = ['williamhill', 'bet365', 'unibet', 'betfair', 'paddypower', 'skybet', 'ladbrokes', 'coral', 'betway', '888sport']
    
    for event in events:
        event_name = event.get('commence_time', 'Unknown Date')
        print(f"\n========================================================")
        print(f"EVENT: {event_name}")
        print(f"========================================================")
        print(f"{'Matchup':<40} | {'Pred':<8} | {'Conf':<6} | {'Odds':<6} | {'Edge':<6} | {'Action':<15}")
        print("-" * 95)
        
        # Find best available odds across UK bookies
        for bookmaker in event['bookmakers']:
            if bookmaker['key'] in uk_bookies: 
                # Process markets
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h':
                        outcomes = market['outcomes']
                        if len(outcomes) != 2: continue
                        
                        f1_api_name = outcomes[0]['name']
                        f2_api_name = outcomes[1]['name']
                        o1 = outcomes[0]['price']
                        o2 = outcomes[1]['price']
                        
                        # Match to DB
                        f1_db = find_fighter_in_db(f1_api_name, db_names)
                        f2_db = find_fighter_in_db(f2_api_name, db_names)
                        
                        if not f1_db or not f2_db:
                            print(f"{f1_api_name} vs {f2_api_name:<30} | {'N/A':<8} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'SKIP (No Data)':<15}")
                            continue
                            
                        # Construct Features
                        X = construct_features(f1_db, f2_db, o1, o2, fighter_db, features)
                        if X is None: continue
                        
                        # Predict
                        # 1. XGB Raw
                        X_xgb = X[[c for c in features if c in X.columns]]
                        p_xgb_raw = xgb_model.predict_proba(X_xgb)[:, 1][0]
                        
                        # 2. Siamese Raw
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
                            
                        # 3. Calibrate
                        p_xgb_iso = iso_xgb.predict([p_xgb_raw])[0]
                        p_siam_iso = iso_siam.predict([p_siam_raw])[0]
                        
                        # 4. Ensemble
                        w = 0.405
                        final_prob = w * p_xgb_iso + (1 - w) * p_siam_iso
                        
                        # 5. Evaluate
                        if final_prob > 0.5:
                            my_prob = final_prob
                            odds = o1
                            fighter = f1_db
                            opponent = f2_db
                        else:
                            my_prob = 1 - final_prob
                            odds = o2
                            fighter = f2_db
                            opponent = f1_db
                            
                        implied = 1.0 / odds
                        edge = my_prob - implied
                        
                        matchup_str = f"{fighter} vs {opponent}"
                        action = "PASS"
                        
                        if my_prob < 0.60: action = "PASS (Low Conf)"
                        elif odds > 5.0: action = "PASS (High Odds)"
                        elif edge <= 0: action = "PASS (No Edge)"
                        else:
                            # Kelly
                            b = odds - 1.0
                            q = 1.0 - my_prob
                            f_star = (b * my_prob - q) / b
                            stake_pct = f_star * 0.25
                            action = f"BET {stake_pct:.1%}"
                            
                            bets_to_make.append({
                                "Fighter": fighter,
                                "Opponent": opponent,
                                "Odds": odds,
                                "Prob": my_prob,
                                "Edge": edge,
                                "Kelly_Stake": stake_pct,
                                "Book": bookmaker['title']
                            })
                        
                        print(f"{matchup_str:<40} | {fighter:<8} | {my_prob:<6.1%} | {odds:<6.2f} | {edge:<6.1%} | {action:<15}")
                            
                break # Only process one bookmaker per fight (first UK one found)
                
    print("\n\n=== SUMMARY OF RECOMMENDED UK BETS ===")
    if not bets_to_make:
        print("No bets found matching criteria.")
    else:
        print(f"{'Fighter':<20} | {'Odds':<6} | {'Prob':<6} | {'Edge':<6} | {'Stake':<6} | {'Book':<10}")
        print("-" * 70)
        for b in bets_to_make:
            print(f"{b['Fighter']:<20} | {b['Odds']:<6.2f} | {b['Prob']:<6.1%} | {b['Edge']:<6.1%} | {b['Kelly_Stake']:<6.1%} | {b['Book']:<10}")

if __name__ == "__main__":
    predict_upcoming()
