import joblib
import json
import pandas as pd
import numpy as np
import difflib
import sys

def load_resources():
    print("Loading resources...")
    try:
        model = joblib.load('ufc_model_optimized.pkl')
        with open('fighter_db.json', 'r') as f:
            fighter_db = json.load(f)
        with open('final_features.json', 'r') as f:
            features = json.load(f)
        return model, fighter_db, features
    except FileNotFoundError as e:
        print(f"Error loading resources: {e}")
        print("Please ensure you have trained the model and built the database.")
        sys.exit(1)

def find_fighter(name, db_names):
    # Exact match
    if name in db_names:
        return name
    
    # Case insensitive
    matches = [n for n in db_names if n.lower() == name.lower()]
    if matches:
        return matches[0]
        
    # Fuzzy match
    close_matches = difflib.get_close_matches(name, db_names, n=3, cutoff=0.6)
    if close_matches:
        print(f"Did you mean: {', '.join(close_matches)}?")
        # Return the best match if it's very close, or ask user?
        # For simplicity, let's ask the user to confirm if multiple, or just pick first if very close.
        # Let's just return the first one for now but warn.
        print(f"Assuming you meant: {close_matches[0]}")
        return close_matches[0]
    
    return None

def get_fighter_stats(name, db):
    return db.get(name)

def construct_features(f1_name, f2_name, f1_odds, f2_odds, db, feature_names):
    stats_1 = get_fighter_stats(f1_name, db)
    stats_2 = get_fighter_stats(f2_name, db)
    
    row = {}
    for feat in feature_names:
        val = np.nan
        
        if feat == 'f_1_odds': val = f1_odds
        elif feat == 'f_2_odds': val = f2_odds
        elif feat == 'diff_odds': val = f1_odds - f2_odds
        
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
            
        elif feat.startswith('diff_'):
            base = feat[5:]
            v1 = stats_1.get(base)
            v2 = stats_2.get(base)
            if v1 is not None and v2 is not None:
                val = v1 - v2
        
        row[feat] = val
        
    return pd.DataFrame([row])

def main():
    model, db, features = load_resources()
    db_names = list(db.keys())
    
    print("\n--- UFC Fight Predictor (Interactive) ---")
    print("Type 'exit' to quit.")
    
    while True:
        print("\n----------------------------------------")
        f1_input = input("Fighter 1 Name: ").strip()
        if f1_input.lower() == 'exit': break
        
        f1_name = find_fighter(f1_input, db_names)
        if not f1_name:
            print(f"Fighter '{f1_input}' not found!")
            continue
            
        f2_input = input("Fighter 2 Name: ").strip()
        if f2_input.lower() == 'exit': break
        
        f2_name = find_fighter(f2_input, db_names)
        if not f2_name:
            print(f"Fighter '{f2_input}' not found!")
            continue
            
        try:
            o1 = float(input(f"Odds for {f1_name} (Decimal): "))
            o2 = float(input(f"Odds for {f2_name} (Decimal): "))
        except ValueError:
            print("Invalid odds! Please enter numbers.")
            continue
            
        # Predict
        try:
            X = construct_features(f1_name, f2_name, o1, o2, db, features)
            probs = model.predict_proba(X)[0]
            
            p_f1 = probs[1]
            p_f2 = probs[0]
            
            print(f"\n>>> Prediction <<<")
            print(f"{f1_name} ({p_f1:.1%}) vs {f2_name} ({p_f2:.1%})")
            
            winner = f1_name if p_f1 > 0.5 else f2_name
            print(f"Winner: {winner}")
            
            # Value Bet Check
            implied_1 = 1/o1
            implied_2 = 1/o2
            
            if p_f1 > implied_1 + 0.05:
                print(f"** VALUE BET ALERT **: {f1_name} (Edge: {p_f1 - implied_1:.1%})")
            elif p_f2 > implied_2 + 0.05:
                print(f"** VALUE BET ALERT **: {f2_name} (Edge: {p_f2 - implied_2:.1%})")
                
        except Exception as e:
            print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
