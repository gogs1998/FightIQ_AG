import pandas as pd
import joblib
import os
import sys
from feature_utils import prepare_production_data

def predict_next_card():
    print("=== Master Props: Predicting Next Card ===")
    
    # 1. Load Upcoming Fights (Mock for now, replace with API fetch later)
    # We need to load the latest data which contains upcoming fights
    try:
        df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    except:
        df = pd.read_csv('data/training_data_enhanced.csv')
        
    # Filter for future fights (or just use the holdout set for demo)
    df['event_date'] = pd.to_datetime(df['event_date'])
    # For demo purposes, let's predict the LAST event in the dataset (most recent)
    last_date = df['event_date'].max()
    print(f"Predicting for event on: {last_date.date()}")
    
    upcoming_df = df[df['event_date'] == last_date].copy()
    
    if upcoming_df.empty:
        print("No upcoming fights found.")
        return

    # 2. Prepare Data
    X_upcoming, _ = prepare_production_data(upcoming_df)
    
    # 3. Load Frozen Models
    models_dir = 'models'
    print("Loading frozen models...")
    model_win = joblib.load(f'{models_dir}/production_winner.pkl')
    model_finish = joblib.load(f'{models_dir}/production_finish.pkl')
    model_method = joblib.load(f'{models_dir}/production_method.pkl')
    model_round = joblib.load(f'{models_dir}/production_round.pkl')
    
    # 4. Predict
    print("Generating predictions...")
    p_win = model_win.predict_proba(X_upcoming)[:, 1]
    p_finish = model_finish.predict_proba(X_upcoming)[:, 1]
    p_method = model_method.predict_proba(X_upcoming)
    p_round = model_round.predict_proba(X_upcoming)
    
    # 5. Build Card
    print("\n=== FIGHT CARD PREDICTIONS (Value Calculator) ===")
    print(f"{'Fight':<40} | {'Winner':<20} | {'Method':<15} | {'Round':<5} | {'Trifecta %':<10} | {'Min Odds':<10}")
    print("-" * 115)
    
    for i in range(len(upcoming_df)):
        row = upcoming_df.iloc[i]
        fight_str = f"{row['f_1_name']} vs {row['f_2_name']}"
        
        # Winner
        prob_w = p_win[i]
        if prob_w > 0.5:
            pred_winner = row['f_1_name']
            conf_w = prob_w
        else:
            pred_winner = row['f_2_name']
            conf_w = 1 - prob_w
            
        # Method
        prob_ko = p_finish[i] * p_method[i][0]
        prob_sub = p_finish[i] * p_method[i][1]
        prob_dec = 1 - p_finish[i]
        
        methods = {'KO/TKO': prob_ko, 'Submission': prob_sub, 'Decision': prob_dec}
        pred_method = max(methods, key=methods.get)
        conf_m = methods[pred_method]
        
        # Round
        import numpy as np
        best_round_idx = np.argmax(p_round[i])
        pred_round = best_round_idx + 1
        conf_r = p_round[i][best_round_idx]
        
        # Trifecta Probability
        if pred_method == 'Decision':
            trifecta_prob = conf_w * prob_dec
            round_str = "-"
        else:
            trifecta_prob = conf_w * conf_m * conf_r
            round_str = str(pred_round)
            
        # Calculate Minimum Profitable Odds (Decimal)
        # EV = (Prob * (Odds - 1)) - (1 - Prob) > 0
        # Prob * Odds - Prob - 1 + Prob > 0
        # Prob * Odds > 1
        # Odds > 1 / Prob
        if trifecta_prob > 0:
            min_odds_dec = 1 / trifecta_prob
            # Convert to American for display
            if min_odds_dec >= 2.0:
                min_odds_us = f"+{int((min_odds_dec - 1) * 100)}"
            else:
                min_odds_us = f"-{int(100 / (min_odds_dec - 1))}"
        else:
            min_odds_us = "N/A"
            
        print(f"{fight_str:<40} | {pred_winner:<20} ({conf_w:.0%}) | {pred_method:<15} | {round_str:<5} | {trifecta_prob:.1%}   | {min_odds_us:<10}")

if __name__ == "__main__":
    predict_next_card()
