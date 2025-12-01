import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import sys
from sklearn.metrics import accuracy_score

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

def run_walk_forward():
    print("=== Unified Props: Walk-Forward Validation (2020-2024) ===")
    
    # 1. Load Data
    print("Loading data...")
    data_path = '../../../master_3/data/training_data_enhanced.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Load Features
    feat_path = '../../../master_3/features_enhanced.json'
    with open(feat_path, 'r') as f:
        features = json.load(f)
        
    # Params (Baseline)
    params = {
        'n_estimators': 100, # Reduced for speed in validation
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1
    }
    
    years = [2020, 2021, 2022, 2023, 2024]
    results = []
    
    for year in years:
        split_date = f'{year}-01-01'
        test_end_date = f'{year+1}-01-01'
        
        print(f"\n--- Validating Year: {year} ---")
        
        # Train: All history BEFORE split_date
        train_mask = df['event_date'] < split_date
        # Test: Only THIS year
        test_mask = (df['event_date'] >= split_date) & (df['event_date'] < test_end_date)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"Train: {len(train_df)} | Test: {len(test_df)}")
        
        if len(test_df) == 0:
            print("Skipping (No test data)")
            continue
            
        X_train = train_df[features].fillna(0)
        X_test = test_df[features].fillna(0)
        
        # --- Train Models ---
        
        # 1. Winner
        y_train_win = train_df['winner_encoded']
        model_win = xgb.XGBClassifier(**params, objective='binary:logistic')
        model_win.fit(X_train, y_train_win)
        p_win = model_win.predict_proba(X_test)[:, 1]
        
        # 2. Finish
        y_train_finish = train_df['is_finish']
        model_finish = xgb.XGBClassifier(**params, objective='binary:logistic')
        model_finish.fit(X_train, y_train_finish)
        p_finish = model_finish.predict_proba(X_test)[:, 1]
        
        # 3. Method (KO vs Sub) - Trained on Finishes Only
        mask_finish_train = train_df['is_finish'] == 1
        X_train_method = X_train[mask_finish_train]
        y_train_method = train_df.loc[mask_finish_train, 'result'].apply(lambda x: 1 if 'Submission' in str(x) else 0)
        
        model_method = xgb.XGBClassifier(**params, objective='binary:logistic')
        model_method.fit(X_train_method, y_train_method)
        p_method = model_method.predict_proba(X_test) # [KO, Sub]
        
        # 4. Round (1-5) - Trained on Finishes Only
        y_train_round = (train_df.loc[mask_finish_train, 'finish_round'].astype(int) - 1).clip(0, 4)
        
        params_round = params.copy()
        model_round = xgb.XGBClassifier(**params_round, objective='multi:softprob', num_class=5)
        model_round.fit(X_train_method, y_train_round)
        p_round = model_round.predict_proba(X_test)
        
        # --- Evaluate ---
        correct_win = 0
        correct_method = 0
        correct_round = 0
        correct_trifecta = 0
        
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            
            # Winner
            pred_winner_idx = 1 if p_win[i] > 0.5 else 0
            
            # Method
            prob_ko = p_finish[i] * p_method[i][0]
            prob_sub = p_finish[i] * p_method[i][1]
            prob_dec = 1 - p_finish[i]
            
            methods = {'KO/TKO': prob_ko, 'Submission': prob_sub, 'Decision': prob_dec}
            pred_method_str = max(methods, key=methods.get)
            
            # Round
            best_rnd_idx = np.argmax(p_round[i])
            pred_round_val = best_rnd_idx + 1
            
            # Actuals
            actual_winner = row['winner_encoded']
            actual_res = str(row['result'])
            try: actual_round = int(row['finish_round'])
            except: actual_round = -1
            
            # Checks
            is_win_correct = (pred_winner_idx == actual_winner)
            
            is_method_correct = False
            if 'KO' in actual_res and pred_method_str == 'KO/TKO': is_method_correct = True
            elif 'Submission' in actual_res and pred_method_str == 'Submission': is_method_correct = True
            elif 'Decision' in actual_res and pred_method_str == 'Decision': is_method_correct = True
            
            is_round_correct = False
            if 'Decision' in actual_res:
                if pred_method_str == 'Decision': is_round_correct = True
            else:
                if pred_method_str != 'Decision' and pred_round_val == actual_round:
                    is_round_correct = True
                    
            if is_win_correct: correct_win += 1
            if is_method_correct: correct_method += 1
            if is_round_correct: correct_round += 1
            if is_win_correct and is_method_correct and is_round_correct:
                correct_trifecta += 1
                
        acc_win = correct_win / len(test_df)
        acc_method = correct_method / len(test_df)
        acc_round = correct_round / len(test_df)
        acc_trifecta = correct_trifecta / len(test_df)
        
        print(f"Year {year}: Win={acc_win:.2%}, Method={acc_method:.2%}, Round={acc_round:.2%}, Trifecta={acc_trifecta:.2%}")
        
        results.append({
            'Year': year,
            'Win_Acc': acc_win,
            'Method_Acc': acc_method,
            'Round_Acc': acc_round,
            'Trifecta_Acc': acc_trifecta,
            'N': len(test_df)
        })
    
    # Summary
    print("\n=== Walk-Forward Summary ===")
    res_df = pd.DataFrame(results)
    print(res_df)
    print(f"\nAvg Win Acc: {res_df['Win_Acc'].mean():.2%}")
    print(f"Avg Method Acc: {res_df['Method_Acc'].mean():.2%}")
    print(f"Avg Round Acc: {res_df['Round_Acc'].mean():.2%}")
    print(f"Avg Trifecta Acc: {res_df['Trifecta_Acc'].mean():.2%}")

if __name__ == "__main__":
    run_walk_forward()
