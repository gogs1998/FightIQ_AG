import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
import matplotlib.pyplot as plt

def evaluate_prop_precision():
    print("=== Prop Hunter: Precision & Profitability Analysis (2024-2025) ===")
    
    # 1. Load Data & Models
    # 1. Load Data & Models
    # Use relative paths or assume we are in master_props
    try:
        df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    except:
        df = pd.read_csv('data/training_data_enhanced.csv')
        
    with open('../master_3/features_selected.json', 'r') as f:
        features = json.load(f)
        
    # Add Prop-Specific Features if they exist and aren't in selected
    prop_features = ['f_1_chin_score', 'f_2_chin_score', 'diff_chin_score']
    for f in prop_features:
        if f not in features:
            features.append(f)
        
    model_finish = joblib.load('model_finish.pkl')
    model_method = joblib.load('model_method.pkl')
    
    # Filter valid odds & time
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Test Set (2024-2025)
    mask_test = df['event_date'] >= '2024-01-01'
    test_df = df[mask_test].copy()
    
    X = test_df[[c for c in features if c in test_df.columns]].fillna(0)
    
    print(f"Evaluating on {len(test_df)} fights...")
    
    # 2. Generate Predictions
    p_finish = model_finish.predict_proba(X)[:, 1]
    # Class 0 = KO, Class 1 = Sub
    p_method_probs = model_method.predict_proba(X)
    p_ko = p_method_probs[:, 0]
    p_sub = p_method_probs[:, 1]
    
    # 3. Analyze KO Precision
    # Strategy: Bet KO if P(Finish) * P(KO|Finish) > Threshold
    
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    ko_results = []
    
    print("\n--- KO Prop Analysis ---")
    print(f"{'Threshold':<10} | {'Bets':<6} | {'Wins':<6} | {'Precision':<10} | {'Min Odds Needed':<15}")
    print("-" * 65)
    
    for thresh in thresholds:
        bets = 0
        wins = 0
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            
            # Combined Prob: P(Finish) * P(KO|Finish)
            # Note: This assumes the fighter WINS. 
            # Ideally: P(Win) * P(Finish) * P(KO|Finish)
            # But let's look at "Method IF they win/finish" for now to check signal quality.
            # Actually, for betting, we need P(Win & KO).
            # We don't have P(Win) loaded here, so let's approximate or just look at P(KO|Finish) accuracy.
            
            # Let's evaluate: If we predict "This fight ends in KO", how often does it?
            # P(Fight Ends in KO) = P(Finish) * P(KO|Finish)
            # (Ignoring who wins for a moment, just "Fight ends by KO")
            
            prob_fight_ko = p_finish[i] * p_ko[i]
            
            if prob_fight_ko > thresh:
                bets += 1
                # Check result
                res = str(row['result']).lower()
                if 'ko' in res or 'tko' in res:
                    wins += 1
                    
        if bets > 0:
            prec = wins / bets
            if prec > 0:
                min_odds = 1.0 / prec
            else:
                min_odds = float('inf')
            print(f"{thresh:<10.2f} | {bets:<6} | {wins:<6} | {prec:<10.1%} | {min_odds:<15.2f}")
        else:
            print(f"{thresh:<10.2f} | 0      | 0      | N/A        | N/A")
            
    # 4. Analyze Sub Precision
    print("\n--- Submission Prop Analysis ---")
    print(f"{'Threshold':<10} | {'Bets':<6} | {'Wins':<6} | {'Precision':<10} | {'Min Odds Needed':<15}")
    print("-" * 65)
    
    for thresh in thresholds:
        bets = 0
        wins = 0
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            
            # P(Fight Ends in Sub) = P(Finish) * (1 - P(KO|Finish))
            prob_fight_sub = p_finish[i] * (1 - p_ko[i])
            
            if prob_fight_sub > thresh:
                bets += 1
                res = str(row['result']).lower()
                if 'submission' in res:
                    wins += 1
                    
        if bets > 0:
            prec = wins / bets
            if prec > 0:
                min_odds = 1.0 / prec
            else:
                min_odds = float('inf')
            print(f"{thresh:<10.2f} | {bets:<6} | {wins:<6} | {prec:<10.1%} | {min_odds:<15.2f}")
        else:
            print(f"{thresh:<10.2f} | 0      | 0      | N/A        | N/A")

if __name__ == "__main__":
    evaluate_prop_precision()
