import pandas as pd
import numpy as np
import joblib
import json
import requests
import xgboost as xgb
from datetime import datetime

# API Config
API_KEY = '74b4ca301791b4b4c6ebe95897ac8673'
SPORT = 'mma_mixed_martial_arts'

def predict_next_event():
    print("=== FightIQ: Next Event Prediction Engine ===")
    
    # 1. Fetch Odds from API
    print("Fetching latest odds from The Odds API...")
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/?apiKey={API_KEY}&regions=us&markets=h2h&oddsFormat=decimal'
    
    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return
        
    if not data or not isinstance(data, list):
        print("No events found or API error.")
        return
        
    print(f"Found {len(data)} upcoming events.")
    
    # Filter for UFC
    ufc_events = [e for e in data if 'UFC' in e['sport_title'] or 'Mixed Martial Arts' in e['sport_title']]
    
    if not ufc_events:
        print("No UFC events found.")
        return
        
    # Take the soonest event
    # Sort by time
    ufc_events.sort(key=lambda x: x['commence_time'])
    next_event = ufc_events[0] # Or group by event?
    
    # Actually, the API returns a list of fights, not grouped by event object usually.
    # We need to process all fights.
    
    print(f"Processing {len(ufc_events)} upcoming fights...")
    
    # 2. Load Models & Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df_hist = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df_hist = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    # Load Models
    model_win = joblib.load(f'{BASE_DIR}/prop_hunter/model_finish.pkl') # Wait, this is finish model. We need Win model.
    # We don't have a saved "Win Model" in prop_hunter. We usually retrain it or use experiment_2 model.
    # Let's use the optimized params to retrain quickly on full history for max accuracy.
    
    print("Retraining Win Model on full history (2010-2025)...")
    with open(f'{BASE_DIR}/prop_hunter/features.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    # Prepare Training Data
    has_odds = (df_hist['f_1_odds'].notna()) & (df_hist['f_2_odds'].notna())
    df_train = df_hist[has_odds].copy()
    X_train = df_train[features].fillna(0)
    y_train = df_train['target']
    
    # Train Win Model
    # Use optimized params
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
        
    clf_win = xgb.XGBClassifier(**params)
    clf_win.fit(X_train, y_train)
    
    # Load Prop Models
    clf_finish = joblib.load(f'{BASE_DIR}/prop_hunter/model_finish.pkl')
    clf_method = joblib.load(f'{BASE_DIR}/prop_hunter/model_method.pkl')
    clf_round = joblib.load(f'{BASE_DIR}/prop_hunter/model_round.pkl')
    
    # 3. Build Features for New Fights
    # We need to construct the feature vector for each upcoming fight.
    # This requires looking up the fighters in our history to get their latest stats.
    
    predictions = []
    
    for fight in ufc_events:
        f1_name = fight['home_team']
        f2_name = fight['away_team']
        
        # Clean names
        def clean(n): return n.strip() # Simple clean
        f1_clean = clean(f1_name)
        f2_clean = clean(f2_name)
        
        # Get Odds
        odds_f1 = 1.91
        odds_f2 = 1.91
        book = fight['bookmakers'][0] if fight['bookmakers'] else None
        if book:
            for m in book['markets']:
                if m['key'] == 'h2h':
                    for o in m['outcomes']:
                        if o['name'] == f1_name: odds_f1 = o['price']
                        if o['name'] == f2_name: odds_f2 = o['price']
                        
        # Lookup Stats
        # We need the MOST RECENT row for each fighter to get their Elo, Age, Reach, etc.
        # This is complex. We need a "Fighter State" lookup.
        # Let's assume we can find them in df_hist.
        
        def get_latest_stats(name):
            # Search in f_1_name or f_2_name
            mask = (df_hist['f_1_name'] == name) | (df_hist['f_2_name'] == name)
            fighter_history = df_hist[mask].sort_values('event_date', ascending=False)
            
            if len(fighter_history) == 0:
                return None
                
            last_fight = fighter_history.iloc[0]
            
            # Extract stats. If he was f_1, take f_1 stats. If f_2, take f_2 stats.
            if last_fight['f_1_name'] == name:
                prefix = 'f_1'
            else:
                prefix = 'f_2'
                
            stats = {}
            # We need to map these to the feature names expected by the model
            # The model expects features like 'diff_elo', 'f_1_age', etc.
            # Wait, the model expects 'f_1_...' and 'f_2_...' features for the CURRENT matchup.
            # So we need to extract raw stats (elo, age, reach, slpm, etc) and then compute diffs.
            
            # List of raw stats we need to construct features
            # This is hard to do perfectly dynamically.
            # Simplified: Just grab Elo, Age, Reach, Odds (we have), and maybe a few others if possible.
            # Or just use 0 for unknown stats (risky).
            
            # Better approach: Use the last known values for everything.
            for col in df_hist.columns:
                if col.startswith(prefix):
                    stat_name = col[4:] # remove 'f_1_'
                    stats[stat_name] = last_fight[col]
                    
            return stats

        s1 = get_latest_stats(f1_clean)
        s2 = get_latest_stats(f2_clean)
        
        if not s1 or not s2:
            print(f"Skipping {f1_clean} vs {f2_clean} (Newcomer/Unknown)")
            continue
            
        # Construct Feature Vector
        row_dict = {}
        
        # 1. Odds
        row_dict['f_1_odds'] = odds_f1
        row_dict['f_2_odds'] = odds_f2
        row_dict['diff_odds'] = odds_f1 - odds_f2
        
        # 2. Stats & Diffs
        # We need to reconstruct the 51 Boruta features.
        # Example: 'diff_elo', 'f_1_age', etc.
        
        # Helper to safely get stat
        def get_val(stats, key, default=0):
            return stats.get(key, default)
            
        # We need to iterate through the REQUIRED features and build them
        for feat in features:
            if feat in ['f_1_odds', 'f_2_odds', 'diff_odds']: continue
            
            if feat.startswith('diff_'):
                base = feat[5:] # e.g. 'elo' from 'diff_elo'
                v1 = get_val(s1, base)
                v2 = get_val(s2, base)
                row_dict[feat] = v1 - v2
            elif feat.startswith('f_1_'):
                base = feat[4:]
                row_dict[feat] = get_val(s1, base)
            elif feat.startswith('f_2_'):
                base = feat[4:]
                row_dict[feat] = get_val(s2, base)
                
        # Create DataFrame
        X_new = pd.DataFrame([row_dict])
        # Ensure all cols exist
        for f in features:
            if f not in X_new.columns: X_new[f] = 0
        X_new = X_new[features] # Reorder
        
        # 4. Predict
        p_win = clf_win.predict_proba(X_new)[0, 1]
        p_finish = clf_finish.predict_proba(X_new)[0, 1]
        p_ko_given_finish = clf_method.predict_proba(X_new)[0, 1] # Class 1 = KO? Need to verify mapping.
        # Usually XGBoost classes are sorted. 0=Sub, 1=KO? Or alphabetical?
        # In training script: target = 1 if 'ko' else 0. So 1=KO.
        
        p_round = clf_round.predict_proba(X_new)[0] # Array of probs for R1, R2, R3...
        
        # Derived Probs
        p_dec = 1 - p_finish
        p_ko = p_finish * p_ko_given_finish
        p_sub = p_finish * (1 - p_ko_given_finish)
        
        # Round Probs (Conditional on Finish)
        # P(R1) = P(Finish) * P(R1|Finish)
        p_r1 = p_finish * p_round[0]
        p_r2 = p_finish * p_round[1]
        p_r3 = p_finish * p_round[2]
        
        # Store
        pred = {
            "Fighter": f1_clean,
            "Opponent": f2_clean,
            "Odds": odds_f1,
            "Win%": p_win,
            "Edge": p_win - (1/odds_f1),
            "Props": {
                "KO": p_ko,
                "Sub": p_sub,
                "Dec": p_dec,
                "R1": p_r1,
                "R2": p_r2
            }
        }
        predictions.append(pred)
        
    # 5. Output Report
    print("\n=== 🔮 UFC PREDICTIONS (Upcoming) ===")
    print(f"{'Fighter':<20} vs {'Opponent':<20} | {'Win%':<6} | {'Edge':<6} | {'KO':<5} | {'Sub':<5} | {'Dec':<5}")
    print("-" * 90)
    
    for p in predictions:
        print(f"{p['Fighter']:<20} vs {p['Opponent']:<20} | {p['Win%']:<6.1%} | {p['Edge']:<+6.1%} | {p['Props']['KO']:<5.1%} | {p['Props']['Sub']:<5.1%} | {p['Props']['Dec']:<5.1%}")
        
    # Save to JSON
    with open(f'{BASE_DIR}/upcoming_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=4)
    print("\nSaved predictions to upcoming_predictions.json")

if __name__ == "__main__":
    predict_next_event()
