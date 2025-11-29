import pandas as pd
import numpy as np
import joblib
import json
import torch
import difflib
from models import SiameseMatchupNet, prepare_siamese_data

def generate_fight_report():
    print("=== Generating FightIQ 'Premium' Analysis Report (Full Card) ===")
    
    # 1. Full Card Matchups (from Scraped Data)
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
        print(f"Error: {e}")
        return

    # Helper to find stats
    def get_stats(name):
        if name in fighter_db: return fighter_db[name]
        matches = difflib.get_close_matches(name, fighter_db.keys(), n=1)
        return fighter_db[matches[0]] if matches else None

    print("\n" + "="*80)
    print("FIGHTIQ PREMIUM CARD ANALYSIS")
    print("="*80)

    for f1_raw, o1, f2_raw, o2 in matchups_decimal:
        # Match names to DB
        s1 = get_stats(f1_raw)
        s2 = get_stats(f2_raw)
        
        if not s1 or not s2:
            continue
            
        # Construct Features
        row = {}
        for k, v in s1.items(): row[f'f_1_{k}'] = v
        for k, v in s2.items(): row[f'f_2_{k}'] = v
        row['f_1_odds'] = o1
        row['f_2_odds'] = o2
        row['diff_odds'] = o1 - o2
        
        for feat in features:
            if feat in row: continue
            val = 0.0
            if feat.startswith('diff_'):
                base = feat[5:]
                v1 = s1.get(base)
                v2 = s2.get(base)
                if v1 is not None and v2 is not None: val = v1 - v2
            row[feat] = val
        for feat in features:
            if feat not in row: row[feat] = 0.0
            
        X = pd.DataFrame([row])
        
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
        
        # Determine Favorite
        if final_prob > 0.5:
            my_prob = final_prob
            odds = o1
            fighter = f1_raw
            opponent = f2_raw
            stats_f = s1
            stats_o = s2
            siam_score = p_siam_iso
        else:
            my_prob = 1 - final_prob
            odds = o2
            fighter = f2_raw
            opponent = f1_raw
            stats_f = s2
            stats_o = s1
            siam_score = 1 - p_siam_iso
            
        implied = 1.0 / odds
        edge = my_prob - implied
        
        print(f"\n>>> {fighter} vs {opponent} <<<")
        
        # Explainability (REAL DATA)
        reasons = []
        
        # 1. Age
        age_f = stats_f.get('age', 0)
        age_o = stats_o.get('age', 0)
        if age_f > 0 and age_o > 0:
            diff = age_o - age_f # Positive means Fighter is younger
            if diff >= 3: reasons.append(f"Youth Advantage: {diff:.1f} years younger ({age_f} vs {age_o})")
            elif diff <= -3: reasons.append(f"Age Disadvantage: {abs(diff):.1f} years older ({age_f} vs {age_o})")
            
        # 2. Reach
        reach_f = stats_f.get('reach_cm', 0)
        reach_o = stats_o.get('reach_cm', 0)
        if reach_f > 0 and reach_o > 0:
            diff = reach_f - reach_o
            if diff >= 5: reasons.append(f"Reach Advantage: +{diff:.0f} cm")
            
        # 3. Elo
        elo_f = stats_f.get('elo', 1500)
        elo_o = stats_o.get('elo', 1500)
        diff = elo_f - elo_o
        if diff >= 50: reasons.append(f"Elo Advantage: +{diff:.0f} points (Higher Rated)")
        elif diff <= -50: reasons.append(f"Elo Disadvantage: {diff:.0f} points (Lower Rated)")
        
        # 4. Siamese (Style)
        if siam_score > 0.55: reasons.append(f"Stylistic Matchup: Siamese Network favors {fighter} ({siam_score:.1%})")
        
        # Print Report
        print(f"• Odds: {odds:.2f} (Implied {implied:.1%})")
        print(f"• Model: {my_prob:.1%} (Edge {edge:+.1%})")
        
        if reasons:
            print("• Key Factors:")
            for r in reasons:
                print(f"  - {r}")
        
        if edge > 0:
            kelly = ((odds-1)*my_prob - (1-my_prob))/(odds-1) * 0.25
            if kelly > 0:
                print(f"💰 RECOMMENDATION: BET {fighter} (Stake {kelly:.1%})")
            else:
                print(f"⚠️ RECOMMENDATION: PASS (Kelly <= 0)")
        else:
            print(f"🛑 RECOMMENDATION: PASS (No Value)")

if __name__ == "__main__":
    generate_fight_report()
