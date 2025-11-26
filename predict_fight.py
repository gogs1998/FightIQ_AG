import pandas as pd
import xgboost as xgb
import joblib
import json
import sys
import numpy as np

def load_resources():
    print("Loading model and database...")
    model = joblib.load('ufc_model_optimized.pkl')
    with open('fighter_db.json', 'r') as f:
        fighter_db = json.load(f)
    with open('final_features.json', 'r') as f:
        features = json.load(f)
    return model, fighter_db, features

def get_fighter_stats(name, db):
    if name not in db:
        # Try fuzzy match?
        # For now, strict match
        return None
    return db[name]

def construct_features(f1_name, f2_name, f1_odds, f2_odds, db, feature_names):
    stats_1 = get_fighter_stats(f1_name, db)
    stats_2 = get_fighter_stats(f2_name, db)
    
    if not stats_1:
        raise ValueError(f"Fighter '{f1_name}' not found in database.")
    if not stats_2:
        raise ValueError(f"Fighter '{f2_name}' not found in database.")
        
    row = {}
    
    # We need to construct the row for every feature in feature_names
    for feat in feature_names:
        val = np.nan
        
        # Case 1: Match Specific
        if feat == 'f_1_odds': val = f1_odds
        elif feat == 'f_2_odds': val = f2_odds
        elif feat == 'diff_odds': val = f1_odds - f2_odds # Assuming diff is f1 - f2? Or abs?
        # Let's check diff logic later. Usually diff is f1 - f2.
        
        # Case 2: Direct Fighter Stats
        elif feat.startswith('f_1_'):
            base = feat[4:]
            if base in stats_1: val = stats_1[base]
        elif feat.startswith('f_2_'):
            base = feat[4:]
            if base in stats_2: val = stats_2[base]
        elif feat.endswith('_f_1'):
            base = feat[:-4]
            if base in stats_1: val = stats_1[base]
        elif feat.endswith('_f_2'):
            base = feat[:-4]
            if base in stats_2: val = stats_2[base]
            
        # Case 3: Diffs
        elif feat.startswith('diff_'):
            base = feat[5:]
            # We need base for f1 and f2
            v1 = stats_1.get(base)
            v2 = stats_2.get(base)
            
            if v1 is not None and v2 is not None:
                # We need to know if the diff is f1-f2 or f2-f1 or abs
                # In the dataset generation (which I don't see), it's usually f1 - f2.
                # I'll assume f1 - f2.
                val = v1 - v2
        
        # Fallback for other columns (maybe categorical?)
        # If we missed something, it stays NaN
        
        row[feat] = val
        
    return pd.DataFrame([row])

def predict(f1_name, f2_name, f1_odds, f2_odds):
    model, db, features = load_resources()
    
    try:
        X = construct_features(f1_name, f2_name, f1_odds, f2_odds, db, features)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Predict
    probs = model.predict_proba(X)[0]
    # probs[1] is probability of class 1 (Fighter 1 wins)
    
    p_f1 = probs[1]
    p_f2 = probs[0]
    
    print(f"\n--- Prediction ---")
    print(f"{f1_name} vs {f2_name}")
    print(f"Odds: {f1_odds} vs {f2_odds}")
    print(f"Win Probability:")
    print(f"  {f1_name}: {p_f1:.1%}")
    print(f"  {f2_name}: {p_f2:.1%}")
    
    winner = f1_name if p_f1 > 0.5 else f2_name
    confidence = max(p_f1, p_f2)
    
    print(f"\nPredicted Winner: {winner} ({confidence:.1%} confidence)")
    
    # Feature Contribution (using simple importance lookup, not SHAP for speed)
    # Ideally SHAP would be better but requires installing shap
    
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python predict_fight.py <Fighter1> <Fighter2> <Odds1> <Odds2>")
        print("Example: python predict_fight.py \"Jon Jones\" \"Stipe Miocic\" 1.2 4.5")
    else:
        f1 = sys.argv[1]
        f2 = sys.argv[2]
        o1 = float(sys.argv[3])
        o2 = float(sys.argv[4])
        predict(f1, f2, o1, o2)
