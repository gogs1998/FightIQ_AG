import pandas as pd
import joblib
import os
import sys
import numpy as np
from feature_utils import prepare_production_data

def validate_production_models():
    print("=== Master Props: Validating Production Models (2024-2025) ===")
    
    # 1. Load Data
    try:
        df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    except:
        df = pd.read_csv('data/training_data_enhanced.csv')
        
    # Filter for Test Set (2024-2025)
    df['event_date'] = pd.to_datetime(df['event_date'])
    test_mask = df['event_date'] >= '2024-01-01'
    test_df_raw = df[test_mask].copy().reset_index(drop=True)
    
    print(f"Test Set Size: {len(test_df_raw)} fights")
    
    # 2. Prepare Data
    X_test, test_df = prepare_production_data(test_df_raw)
    
    # 3. Load Frozen Models
    models_dir = 'models'
    print("Loading frozen models...")
    try:
        model_win = joblib.load(f'{models_dir}/production_winner.pkl')
        model_finish = joblib.load(f'{models_dir}/production_finish.pkl')
        model_method = joblib.load(f'{models_dir}/production_method.pkl')
        model_round = joblib.load(f'{models_dir}/production_round.pkl')
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 4. Predict
    print("Generating predictions...")
    p_win = model_win.predict_proba(X_test)[:, 1]
    p_finish = model_finish.predict_proba(X_test)[:, 1]
    p_method = model_method.predict_proba(X_test)
    p_round = model_round.predict_proba(X_test)
    
    # 5. Calculate Metrics
    correct_win = 0
    correct_method = 0
    correct_round = 0
    correct_trifecta = 0
    
    results = []
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # Winner
        prob_w = p_win[i]
        if prob_w > 0.5:
            pred_winner = row['f_1_name']
        else:
            pred_winner = row['f_2_name']
            
        # Method
        prob_ko = p_finish[i] * p_method[i][0]
        prob_sub = p_finish[i] * p_method[i][1]
        prob_dec = 1 - p_finish[i]
        
        methods = {'KO/TKO': prob_ko, 'Submission': prob_sub, 'Decision': prob_dec}
        pred_method = max(methods, key=methods.get)
        
        # Round
        best_round_idx = np.argmax(p_round[i])
        pred_round = best_round_idx + 1
        
        # Actuals
        actual_winner = str(row.get('winner', 'Unknown'))
        actual_res = str(row['result']).lower()
        
        try:
            actual_round = int(row['finish_round'])
        except:
            actual_round = -1
            
        # Check
        is_win_correct = (pred_winner == actual_winner)
        
        is_method_correct = False
        if 'ko' in actual_res and pred_method == 'KO/TKO': is_method_correct = True
        if 'submission' in actual_res and pred_method == 'Submission': is_method_correct = True
        if 'decision' in actual_res and pred_method == 'Decision': is_method_correct = True
        
        is_round_correct = (pred_round == actual_round)
        
        # Trifecta Logic
        # If Method is Decision, Round is ignored (or implicitly correct if it went distance)
        # But for strict Trifecta: Win + Method + Round
        # Let's stick to the strict definition used before
        is_trifecta = is_win_correct and is_method_correct and is_round_correct
        
        if is_win_correct: correct_win += 1
        if is_method_correct: correct_method += 1
        if is_round_correct: correct_round += 1
        if is_trifecta: correct_trifecta += 1
        
        results.append({
            'Fight': f"{row['f_1_name']} vs {row['f_2_name']}",
            'Correct_Trifecta': is_trifecta,
            'Weight_Class': row.get('weight_class', 'Unknown')
        })
        
    n = len(test_df)
    print("\n=== Production Model Metrics (2024-2025) ===")
    print(f"Winner Accuracy:   {correct_win/n:.2%}")
    print(f"Method Accuracy:   {correct_method/n:.2%}")
    print(f"Round Accuracy:    {correct_round/n:.2%}")
    print(f"Trifecta Accuracy: {correct_trifecta/n:.2%}")
    
    # Save for Deep Dive
    pd.DataFrame(results).to_csv('production_validation_results.csv', index=False)

if __name__ == "__main__":
    validate_production_models()
