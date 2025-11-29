import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from datetime import datetime

def predict_ufc_323():
    # 1. Define Fight Card (From Screenshots)
    fights = [
        ("Merab Dvalishvili", 1.22, "Petr Yan", 4.50),
        ("Alexandre Pantoja", 1.44, "Joshua Van", 2.87),
        ("Brandon Moreno", 2.00, "Tatsuro Taira", 1.80),
        ("Henry Cejudo", 3.25, "Payton Talbott", 1.36),
        ("Jan Blachowicz", 1.72, "Bogdan Guskov", 2.10),
        ("Grant Dawson", 1.47, "Manuel Torres", 2.75),
        ("Terrance McKinney", 2.62, "Chris Duncan", 1.53),
        ("Maycee Barber", 1.62, "Karine Silva", 2.30),
        ("Nazim Sadykhov", 2.10, "Fares Ziam", 1.72),
        ("Marvin Vettori", 1.91, "Brunno Ferreira", 1.91),
        ("Edson Barboza", 3.00, "Jalin Turner", 1.40),
        ("Iwo Baraniewski", 1.50, "Ibo Aslan", 2.70),
        ("Mansur Abdul-Malik", 1.10, "Antonio Trocoli", 7.00),
        ("Muhammad Naimov", 3.00, "Mairon Santos", 1.40)
    ]
    
    # 2. Load Data & Retrain Model (Full History)
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df_hist = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df_hist = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    with open(f'{BASE_DIR}/prop_hunter/features.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    # Prepare Training Data
    has_odds = (df_hist['f_1_odds'].notna()) & (df_hist['f_2_odds'].notna())
    df_train = df_hist[has_odds].copy()
    X_train = df_train[features].fillna(0)
    y_train = df_train['target']
    
    # Train Win Model (Optimized Params)
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
        
    clf_win = xgb.XGBClassifier(**params)
    clf_win.fit(X_train, y_train)
    
    # Load Prop Models (Pre-trained)
    clf_finish = joblib.load(f'{BASE_DIR}/prop_hunter/model_finish.pkl')
    clf_method = joblib.load(f'{BASE_DIR}/prop_hunter/model_method.pkl')
    clf_round = joblib.load(f'{BASE_DIR}/prop_hunter/model_round.pkl')
    
    # 3. Predict Each Fight
    predictions = []
    
    for f1_name, odds_f1, f2_name, odds_f2 in fights:
        # Lookup Stats
        def get_latest_stats(name):
            # Try exact match first
            mask = (df_hist['f_1_name'] == name) | (df_hist['f_2_name'] == name)
            if mask.sum() == 0:
                # Try partial match?
                return None, None
            
            fighter_history = df_hist[mask].sort_values('event_date', ascending=False)
            last_fight = fighter_history.iloc[0]
            last_date = last_fight['event_date']
            
            if last_fight['f_1_name'] == name: prefix = 'f_1'
            else: prefix = 'f_2'
                
            stats = {}
            for col in df_hist.columns:
                if col.startswith(prefix):
                    stat_name = col[4:]
                    stats[stat_name] = last_fight[col]
            return stats, last_date

        s1, d1 = get_latest_stats(f1_name)
        s2, d2 = get_latest_stats(f2_name)
        
        if not s1 or not s2:
            continue
            
        # Construct Feature Vector
        row_dict = {}
        row_dict['f_1_odds'] = odds_f1
        row_dict['f_2_odds'] = odds_f2
        row_dict['diff_odds'] = odds_f1 - odds_f2
        
        def get_val(stats, key, default=0): return stats.get(key, default)
            
        for feat in features:
            if feat in ['f_1_odds', 'f_2_odds', 'diff_odds']: continue
            
            if feat.startswith('diff_'):
                base = feat[5:]
                v1 = get_val(s1, base)
                v2 = get_val(s2, base)
                row_dict[feat] = v1 - v2
            elif feat.startswith('f_1_'):
                base = feat[4:]
                row_dict[feat] = get_val(s1, base)
            elif feat.startswith('f_2_'):
                base = feat[4:]
                row_dict[feat] = get_val(s2, base)
                
        X_new = pd.DataFrame([row_dict])
        for f in features:
            if f not in X_new.columns: X_new[f] = 0
        X_new = X_new[features]
        
        # Predict
        p_win = clf_win.predict_proba(X_new)[0, 1]
        p_finish = clf_finish.predict_proba(X_new)[0, 1]
        p_ko_given_finish = clf_method.predict_proba(X_new)[0, 1]
        p_round = clf_round.predict_proba(X_new)[0]
        
        # Derived
        p_dec = 1 - p_finish
        p_ko = p_finish * p_ko_given_finish
        p_sub = p_finish * (1 - p_ko_given_finish)
        
        # Edge Calculation
        if p_win > 0.5:
            my_prob = p_win
            my_odds = odds_f1
            pick = f1_name
        else:
            my_prob = 1 - p_win
            my_odds = odds_f2
            pick = f2_name
            
        edge = my_prob - (1/my_odds)
        
        pred = {
            "Fighter A": f1_name,
            "Fighter B": f2_name,
            "Pick": pick,
            "Conf": my_prob,
            "Odds": my_odds,
            "Edge": edge,
            "Props": {
                "KO": p_ko,
                "Sub": p_sub,
                "Dec": p_dec,
                "R1": p_finish * p_round[0],
                "R2": p_finish * p_round[1]
            }
        }
        predictions.append(pred)
        
    # Output Table
    print("\n=== 🔮 UFC 323 PREDICTIONS ===")
    for p in predictions:
        matchup = f"{p['Fighter A']} vs {p['Fighter B']}"
        props = p['Props']
        best_method = max(props, key=lambda k: props[k] if k in ['KO', 'Sub', 'Dec'] else 0)
        method_str = f"{best_method} ({props[best_method]:.0%})"
        edge_str = f"{p['Edge']:+.1%}" if p['Edge'] > 0 else "None"
        print(f"{matchup:<40} | {p['Pick']:<20} | {p['Conf']:<6.1%} | {p['Odds']:<5.2f} | {edge_str:<6} | {method_str:<15}")
        
    # Save to JSON (Handle numpy types)
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
        
    with open(f'{BASE_DIR}/ufc_323_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=4, default=convert)
    print("\nSaved predictions to ufc_323_predictions.json")

if __name__ == "__main__":
    predict_ufc_323()
