import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb

def evaluate_round_precision():
    print("=== Prop Hunter: Round/Over-Under Analysis (2024-2025) ===")
    
    # 1. Load Data & Models
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    with open(f'{BASE_DIR}/prop_hunter/features.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    model_finish = joblib.load(f'{BASE_DIR}/prop_hunter/model_finish.pkl')
    
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
    # P(Finish) = Probability fight ends inside distance (Under 2.5 / Under 4.5 usually correlates)
    # P(Decision) = 1 - P(Finish) (Over 1.5 / Over 2.5 usually correlates)
    
    p_finish = model_finish.predict_proba(X)[:, 1]
    
    # 3. Analyze "Goes The Distance" (GTD) vs "Inside The Distance" (ITD)
    # GTD is roughly equivalent to "Over 2.5" in many cases, or exactly "Will the fight go the distance?"
    
    thresholds = [0.50, 0.60, 0.70, 0.80]
    
    print("\n--- 'Inside the Distance' (ITD / Finish) Analysis ---")
    print(f"{'Threshold':<10} | {'Bets':<6} | {'Wins':<6} | {'Precision':<10} | {'Min Odds Needed':<15}")
    print("-" * 65)
    
    for thresh in thresholds:
        bets = 0
        wins = 0
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            prob = p_finish[i]
            
            if prob > thresh:
                bets += 1
                # Check result
                res = str(row['result']).lower()
                if 'decision' not in res: # It was a finish
                    wins += 1
                    
        if bets > 0:
            prec = wins / bets
            min_odds = 1.0 / prec
            print(f"{thresh:<10.2f} | {bets:<6} | {wins:<6} | {prec:<10.1%} | {min_odds:<15.2f}")
        else:
            print(f"{thresh:<10.2f} | 0      | 0      | N/A        | N/A")
            
    print("\n--- 'Goes the Distance' (GTD / Decision) Analysis ---")
    print(f"{'Threshold':<10} | {'Bets':<6} | {'Wins':<6} | {'Precision':<10} | {'Min Odds Needed':<15}")
    print("-" * 65)
    
    for thresh in thresholds:
        bets = 0
        wins = 0
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            prob_gtd = 1 - p_finish[i]
            
            if prob_gtd > thresh:
                bets += 1
                res = str(row['result']).lower()
                if 'decision' in res: # It went to decision
                    wins += 1
                    
        if bets > 0:
            prec = wins / bets
            min_odds = 1.0 / prec
            print(f"{thresh:<10.2f} | {bets:<6} | {wins:<6} | {prec:<10.1%} | {min_odds:<15.2f}")
        else:
            print(f"{thresh:<10.2f} | 0      | 0      | N/A        | N/A")

if __name__ == "__main__":
    evaluate_round_precision()
