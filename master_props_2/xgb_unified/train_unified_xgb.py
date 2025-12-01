import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import sys
from sklearn.metrics import accuracy_score, log_loss

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'master_3')))

# Import adjustment if needed (though we train on raw+enhanced, usually adjustment is for inference)
# But wait, Master 3 Winner model was trained on ADJUSTED features?
# Let's check master_3/train.py if possible.
# Usually, we train on historical data which IS the "adjusted" reality of that time?
# No, historical data contains raw stats entering the fight.
# Adjustment is usually applied to create "opponent-adjusted" stats.
# If we want to use adjusted stats, we must apply adjustment to the training set too?
# master_3/train.py usually does this.
# Let's assume we should use the features as they are in training_data_enhanced.csv
# BUT, if training_data_enhanced.csv does NOT have _adj columns, and we want to use them, we must generate them.
# For this baseline, let's stick to the features PRESENT in training_data_enhanced.csv + features_enhanced.json.
# If features_enhanced.json contains _adj, we need to generate them.
# Let's check features_enhanced.json content first.

def train_unified_xgb():
    print("=== Experiment A: Unified XGB Pipeline ===")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('../../master_3/data/training_data_enhanced.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Load Features
    with open('../../master_3/features_enhanced.json', 'r') as f:
        features = json.load(f)
        
    print(f"Total Fights: {len(df)}")
    print(f"Feature Count: {len(features)}")
    
    # Check for _adj in features
    adj_feats = [f for f in features if '_adj' in f]
    if adj_feats:
        print(f"Warning: {len(adj_feats)} adjusted features found in feature list.")
        # Check if they are in df
        missing_adj = [f for f in adj_feats if f not in df.columns]
        if missing_adj:
            print(f"Generating {len(missing_adj)} missing adjusted features...")
            from master_3.models.opponent_adjustment import apply_opponent_adjustment
            stat_cols = [
                'slpm_15_f_1', 'sapm_15_f_1', 'td_avg_15_f_1', 'sub_avg_15_f_1',
                'slpm_15_f_2', 'sapm_15_f_2', 'td_avg_15_f_2', 'sub_avg_15_f_2'
            ]
            # Only use cols that exist
            stat_cols = [c for c in stat_cols if c in df.columns]
            df = apply_opponent_adjustment(df, stat_cols, elo_col='dynamic_elo')
            
    # 2. Split Data (Time-based)
    train_mask = df['event_date'] < '2024-01-01'
    test_mask = df['event_date'] >= '2024-01-01'
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Train Set: {len(train_df)}")
    print(f"Test Set:  {len(test_df)}")
    
    X_train = train_df[features].fillna(0)
    y_train_win = train_df['winner_encoded']
    
    X_test = test_df[features].fillna(0)
    y_test_win = test_df['winner_encoded']
    
    # 3. Train Winner Model
    print("\nTraining Winner Model...")
    # Using generic params for baseline, or master_3 params if available
    params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': -1
    }
    
    model_win = xgb.XGBClassifier(**params)
    model_win.fit(X_train, y_train_win)
    
    # Evaluate Winner
    p_win = model_win.predict_proba(X_test)[:, 1]
    acc_win = accuracy_score(y_test_win, (p_win > 0.5).astype(int))
    print(f"Winner Accuracy: {acc_win:.2%}")
    
    # 4. Train Prop Models
    # Filter for finishes for training props?
    # Finish Model: All fights (Finish=1, Decision=0)
    print("\nTraining Finish Model...")
    y_train_finish = train_df['is_finish']
    model_finish = xgb.XGBClassifier(**params)
    model_finish.fit(X_train, y_train_finish)
    
    # Method Model: Only Finishes (KO=0, Sub=1)
    print("Training Method Model (KO vs Sub)...")
    mask_finish_train = train_df['is_finish'] == 1
    X_train_method = X_train[mask_finish_train]
    # Encode method: KO/TKO -> 0, Submission -> 1
    # We need to ensure 'result' column is clean
    # Assuming 'result' contains 'KO/TKO' or 'Submission'
    # Or use 'exact_method' if available?
    # Let's check unique values
    # For now, simple logic:
    y_train_method = train_df.loc[mask_finish_train, 'result'].apply(
        lambda x: 1 if 'Submission' in str(x) else 0
    )
    
    model_method = xgb.XGBClassifier(**params)
    model_method.fit(X_train_method, y_train_method)
    
    # Round Model: Only Finishes (1-5)
    print("Training Round Model (1-5)...")
    # rounds are 1-5. XGB needs 0-4
    y_train_round = train_df.loc[mask_finish_train, 'finish_round'].astype(int) - 1
    # Clip to 0-4 just in case
    y_train_round = y_train_round.clip(0, 4)
    
    # Remove objective from params for multi-class
    params_round = params.copy()
    if 'objective' in params_round:
        del params_round['objective']
        
    model_round = xgb.XGBClassifier(**params_round, objective='multi:softprob', num_class=5)
    model_round.fit(X_train_method, y_train_round)
    
    # 5. Save Models
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_win, 'models/xgb_unified_winner.pkl')
    joblib.dump(model_finish, 'models/xgb_unified_finish.pkl')
    joblib.dump(model_method, 'models/xgb_unified_method.pkl')
    joblib.dump(model_round, 'models/xgb_unified_round.pkl')
    print("\nModels saved to master_props_2/xgb_unified/models/")
    
    # 6. Validate Trifecta
    print("\n=== Validation (Trifecta) ===")
    
    # Predict Props on Test
    p_finish = model_finish.predict_proba(X_test)[:, 1]
    p_method = model_method.predict_proba(X_test) # [KO, Sub]
    p_round = model_round.predict_proba(X_test)   # [R1..R5]
    
    correct_trifecta = 0
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # Winner
        if p_win[i] > 0.5:
            pred_winner_idx = 1 # f_1
            pred_winner_name = row['f_1_name']
        else:
            pred_winner_idx = 0 # f_2
            pred_winner_name = row['f_2_name']
            
        # Method Logic
        # P(KO) = P(Finish) * P(Method=KO|Finish)
        # P(Sub) = P(Finish) * P(Method=Sub|Finish)
        # P(Dec) = 1 - P(Finish)
        
        prob_ko = p_finish[i] * p_method[i][0]
        prob_sub = p_finish[i] * p_method[i][1]
        prob_dec = 1 - p_finish[i]
        
        methods = {'KO/TKO': prob_ko, 'Submission': prob_sub, 'Decision': prob_dec}
        pred_method_str = max(methods, key=methods.get)
        
        # Round Logic
        best_rnd_idx = np.argmax(p_round[i])
        pred_round_val = best_rnd_idx + 1
        
        # Actuals
        actual_winner = row['winner_encoded'] # 1 or 0
        actual_res = str(row['result'])
        try:
            actual_round = int(row['finish_round'])
        except:
            actual_round = -1
            
        # Check
        is_win_correct = (pred_winner_idx == actual_winner)
        
        is_method_correct = False
        if 'KO' in actual_res and pred_method_str == 'KO/TKO': is_method_correct = True
        if 'Submission' in actual_res and pred_method_str == 'Submission': is_method_correct = True
        if 'Decision' in actual_res and pred_method_str == 'Decision': is_method_correct = True
        
        is_round_correct = (pred_round_val == actual_round)
        
        # Trifecta
        if pred_method_str == 'Decision':
            is_trifecta = is_win_correct and is_method_correct
        else:
            is_trifecta = is_win_correct and is_method_correct and is_round_correct
            
        if is_trifecta:
            correct_trifecta += 1
            
    acc_trifecta = correct_trifecta / len(test_df)
    print(f"Trifecta Accuracy: {acc_trifecta:.2%}")

if __name__ == "__main__":
    train_unified_xgb()
