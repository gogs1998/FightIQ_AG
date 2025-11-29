import pandas as pd
import numpy as np
import joblib
import json
import torch
import difflib
from models import SiameseMatchupNet, prepare_siamese_data

def analyze_scraped_odds():
    print("=== FightIQ: Analyzing Scraped Odds (UFC 323) ===")
    
    # 1. Manual Data Entry (from Bet365 Screenshots)
    # Format: (Fighter1, Odds1_Decimal, Fighter2, Odds2_Decimal)
    
    matchups_decimal = [
        ("Muhammad Naimov", 3.00, "Mairon Santos", 1.40),
        ("Mansur Abdul-Malik", 1.10, "Antonio Trocoli", 7.00),
        ("Iwo Baraniewski", 1.50, "Ibo Aslan", 2.70),
        ("Edson Barboza", 3.00, "Jalin Turner", 1.40),
        ("Marvin Vettori", 1.91, "Brunno Ferreira", 1.91),
        ("Nazim Sadykhov", 2.10, "Fares Ziam", 1.73),
        ("Maycee Barber", 1.625, "Karine Silva", 2.30),
        ("Terrance McKinney", 2.625, "Chris Duncan", 1.53),
        ("Grant Dawson", 1.48, "Manuel Torres", 2.75),
        ("Jan Blachowicz", 1.73, "Bogdan Guskov", 2.10),
        ("Henry Cejudo", 3.25, "Payton Talbott", 1.36),
        ("Brandon Moreno", 2.00, "Tatsuro Taira", 1.80),
        ("Alexandre Pantoja", 1.44, "Joshua Van", 2.875),
        ("Merab Dvalishvili", 1.22, "Petr Yan", 4.50)
    ]
    
    # 2. Load Resources
    print("Loading models and database...")
    try:
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
        
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    # Helper Functions
    def find_fighter_in_db(name, db_names):
        if name in db_names: return name
        matches = difflib.get_close_matches(name, db_names, n=1, cutoff=0.6)
        if matches: return matches[0]
        return None

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
        for feat in feature_names:
            if feat not in row: row[feat] = 0.0
        return pd.DataFrame([row])

    print(f"\n{'Matchup':<40} | {'Pred':<8} | {'Conf':<6} | {'Odds':<6} | {'Edge':<6} | {'Action':<15}")
    print("-" * 95)

    db_names = list(fighter_db.keys())
    
    for f1_raw, o1_dec, f2_raw, o2_dec in matchups_decimal:
        f1_db = find_fighter_in_db(f1_raw, db_names)
        f2_db = find_fighter_in_db(f2_raw, db_names)
        
        if not f1_db or not f2_db:
            print(f"{f1_raw} vs {f2_raw:<30} | {'N/A':<8} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'SKIP (No Data)':<15}")
            continue
            
        X = construct_features(f1_db, f2_db, o1_dec, o2_dec, fighter_db, features)
        if X is None: continue
        
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
        
        if final_prob > 0.5:
            my_prob = final_prob
            odds = o1_dec
            fighter = f1_db
            opponent = f2_db
        else:
            my_prob = 1 - final_prob
            odds = o2_dec
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
            b = odds - 1.0
            q = 1.0 - my_prob
            f_star = (b * my_prob - q) / b
            stake_pct = f_star * 0.25
            action = f"BET {stake_pct:.1%}"
        
        print(f"{matchup_str:<40} | {fighter:<8} | {my_prob:<6.1%} | {odds:<6.2f} | {edge:<6.1%} | {action:<15}")

if __name__ == "__main__":
    analyze_scraped_odds()
