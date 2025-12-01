import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sklearn.metrics import accuracy_score

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def validate_metrics():
    print("=== Calculating Detailed Metrics (Experiment A) ===")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('../../master_3/data/training_data_enhanced.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Load Features
    with open('../../master_3/features_enhanced.json', 'r') as f:
        features = json.load(f)
        
    # 2. Split Data (Test Set Only)
    test_mask = df['event_date'] >= '2024-01-01'
    test_df = df[test_mask].copy()
    print(f"Test Set: {len(test_df)} fights (2024-2025)")
    
    X_test = test_df[features].fillna(0)
    y_test_win = test_df['winner_encoded']
    
    # 3. Load Models
    print("Loading models...")
    models_dir = 'models'
    model_win = joblib.load(os.path.join(models_dir, 'xgb_unified_winner.pkl'))
    model_finish = joblib.load(os.path.join(models_dir, 'xgb_unified_finish.pkl'))
    model_method = joblib.load(os.path.join(models_dir, 'xgb_unified_method.pkl'))
    model_round = joblib.load(os.path.join(models_dir, 'xgb_unified_round.pkl'))
    
    # 4. Predict
    print("Running predictions...")
    p_win = model_win.predict_proba(X_test)[:, 1]
    p_finish = model_finish.predict_proba(X_test)[:, 1]
    p_method = model_method.predict_proba(X_test) # [KO, Sub]
    p_round = model_round.predict_proba(X_test)   # [R1..R5]
    
    # 5. Calculate Metrics
    correct_win = 0
    correct_method = 0
    correct_round = 0
    correct_trifecta = 0
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # --- Prediction Logic ---
        
        # Winner
        if p_win[i] > 0.5:
            pred_winner_idx = 1
        else:
            pred_winner_idx = 0
            
        # Method
        prob_ko = p_finish[i] * p_method[i][0]
        prob_sub = p_finish[i] * p_method[i][1]
        prob_dec = 1 - p_finish[i]
        
        methods = {'KO/TKO': prob_ko, 'Submission': prob_sub, 'Decision': prob_dec}
        pred_method_str = max(methods, key=methods.get)
        
        # Round
        best_rnd_idx = np.argmax(p_round[i])
        pred_round_val = best_rnd_idx + 1
        
        # --- Actuals ---
        actual_winner = row['winner_encoded']
        actual_res = str(row['result'])
        try:
            actual_round = int(row['finish_round'])
        except:
            actual_round = -1
            
        # --- Check ---
        
        # 1. Winner Accuracy
        is_win_correct = (pred_winner_idx == actual_winner)
        if is_win_correct: correct_win += 1
        
        # 2. Method Accuracy
        # We check if the predicted method matches the actual method string
        is_method_correct = False
        if 'KO' in actual_res and pred_method_str == 'KO/TKO': is_method_correct = True
        elif 'Submission' in actual_res and pred_method_str == 'Submission': is_method_correct = True
        elif 'Decision' in actual_res and pred_method_str == 'Decision': is_method_correct = True
        
        if is_method_correct: correct_method += 1
        
        # 3. Round Accuracy
        # If Decision, Round is N/A (or we can say it's "Round 3/5" but usually we just say "Decision")
        # For this metric, let's define "Round Accuracy" as:
        # - If it went to decision, did we predict Decision? (Implicitly correct round)
        # - If it was a finish, did we predict the correct Round Number?
        # OR simpler: Did we predict the exact outcome of "Round X" or "Decision"?
        
        is_round_correct = False
        if 'Decision' in actual_res:
            if pred_method_str == 'Decision': is_round_correct = True
        else:
            # It was a finish
            if pred_method_str != 'Decision' and pred_round_val == actual_round:
                is_round_correct = True
                
        if is_round_correct: correct_round += 1
        
        # 4. Trifecta Accuracy
        if is_win_correct and is_method_correct and is_round_correct:
            correct_trifecta += 1
            
    n = len(test_df)
    print("\n=== Final Metrics (2024-2025) ===")
    print(f"Total Fights: {n}")
    print(f"Winner Accuracy:   {correct_win/n:.2%} ({correct_win}/{n})")
    print(f"Method Accuracy:   {correct_method/n:.2%} ({correct_method}/{n})")
    print(f"Round Accuracy:    {correct_round/n:.2%} ({correct_round}/{n})")
    print(f"Trifecta Accuracy: {correct_trifecta/n:.2%} ({correct_trifecta}/{n})")

if __name__ == "__main__":
    validate_metrics()
