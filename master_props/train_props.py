import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.calibration import CalibratedClassifierCV

def train_props():
    print("=== Master Props: Training Hierarchical Models ===")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    
    # Load Features
    with open('../master_3/features_selected.json', 'r') as f:
        features = json.load(f)
        
    # Add Prop-Specific Features if they exist and aren't in selected
    prop_features = ['f_1_chin_score', 'f_2_chin_score', 'diff_chin_score']
    for f in prop_features:
        if f in df.columns and f not in features:
            features.append(f)
            print(f"Added prop feature: {f}")
            
    print(f"Total Features: {len(features)}")
    
    # 2. Prepare Targets
    # Finish: 1 if not Decision, 0 if Decision
    df['is_finish'] = df['result'].apply(lambda x: 0 if 'Decision' in str(x) else 1)
    
    # Method: 0 = KO/TKO, 1 = Submission (NaN if Decision)
    def get_method_target(res):
        res = str(res).lower()
        if 'decision' in res: return np.nan
        if 'ko' in res or 'tko' in res: return 0
        if 'submission' in res: return 1
        return np.nan # DQ/Draw/etc
        
    df['method_target'] = df['result'].apply(get_method_target)
    
    # Round: Exact Round (1-5)
    # If Decision, use 3 (for 3 rnd fight) or 5 (for 5 rnd fight)? 
    # Actually, for "Over/Under", we usually care about the round it ENDED.
    # If decision, it ended in the last round (3 or 5).
    # But for "Round Betting", you bet on "Round 3". If it goes to decision, you lose.
    # So we should train on ALL fights, but maybe treat Decision as a separate class or just max rounds?
    # Let's stick to the plan: Train Round Model on ALL fights.
    # Target: Round Number.
    def get_round_target(row):
        r = row['finish_round']
        if pd.isna(r): return np.nan
        return int(r)
    
    df['round_target'] = df.apply(get_round_target, axis=1)
    
    # Split Data (Time-based)
    df['event_date'] = pd.to_datetime(df['event_date'])
    train_mask = df['event_date'] < '2024-01-01'
    test_mask = df['event_date'] >= '2024-01-01'
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Train Set: {len(train_df)}")
    print(f"Test Set: {len(test_df)}")
    
    # ==========================================
    # Model 1: Finish Model (GTD vs ITD)
    # ==========================================
    print("\n--- Training Finish Model ---")
    X_train = train_df[features].fillna(0)
    y_train = train_df['is_finish']
    X_test = test_df[features].fillna(0)
    y_test = test_df['is_finish']
    
    clf_finish = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Calibrate
    cal_finish = CalibratedClassifierCV(clf_finish, method='isotonic', cv=3)
    cal_finish.fit(X_train, y_train)
    
    p_finish = cal_finish.predict_proba(X_test)[:, 1]
    acc_finish = accuracy_score(y_test, p_finish > 0.5)
    ll_finish = log_loss(y_test, p_finish)
    
    print(f"Finish Model Accuracy: {acc_finish:.2%}")
    print(f"Finish Model LogLoss: {ll_finish:.4f}")
    
    joblib.dump(cal_finish, 'model_finish.pkl')
    
    # ==========================================
    # Model 2: Method Model (KO vs Sub)
    # ==========================================
    print("\n--- Training Method Model (Conditional) ---")
    # Only train on fights that finished
    mask_finish_train = train_df['method_target'].notna()
    X_train_method = train_df[mask_finish_train][features].fillna(0)
    y_train_method = train_df[mask_finish_train]['method_target']
    
    mask_finish_test = test_df['method_target'].notna()
    X_test_method = test_df[mask_finish_test][features].fillna(0)
    y_test_method = test_df[mask_finish_test]['method_target']
    
    clf_method = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    cal_method = CalibratedClassifierCV(clf_method, method='isotonic', cv=3)
    cal_method.fit(X_train_method, y_train_method)
    
    p_method = cal_method.predict_proba(X_test_method)[:, 1] # Prob of Sub
    pred_method = (p_method > 0.5).astype(int)
    
    acc_method = accuracy_score(y_test_method, pred_method)
    ll_method = log_loss(y_test_method, p_method)
    
    print(f"Method Model Accuracy (on Finishes): {acc_method:.2%}")
    print(f"Method Model LogLoss: {ll_method:.4f}")
    
    # Check KO Recall specifically
    print(classification_report(y_test_method, pred_method, target_names=['KO', 'Sub']))
    
    joblib.dump(cal_method, 'model_method.pkl')
    
    # ==========================================
    # Model 3: Round Model (Multi-Class)
    # ==========================================
    print("\n--- Training Round Model ---")
    # Target: 0=R1, 1=R2, 2=R3, 3=R4, 4=R5
    # We need to map rounds to 0-indexed classes
    # Round 1 -> 0, Round 2 -> 1, ...
    
    def prep_round_target(r):
        if pd.isna(r): return np.nan
        r = int(r)
        if r > 5: r = 5
        return r - 1
        
    y_train_round = train_df['round_target'].apply(prep_round_target)
    y_test_round = test_df['round_target'].apply(prep_round_target)
    
    # Filter valid rounds
    mask_r_train = y_train_round.notna()
    X_train_round = train_df[mask_r_train][features].fillna(0)
    y_train_round = y_train_round[mask_r_train]
    
    mask_r_test = y_test_round.notna()
    X_test_round = test_df[mask_r_test][features].fillna(0)
    y_test_round = y_test_round[mask_r_test]
    
    clf_round = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        objective='multi:softprob',
        num_class=5,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # CalibratedCV doesn't support multi-class isotonic easily out of box in older sklearn, 
    # but sigmoid works. Or just use raw XGB probabilities for now.
    # Let's stick to raw XGB for multi-class round prediction to keep it simple.
    clf_round.fit(X_train_round, y_train_round)
    
    p_round = clf_round.predict_proba(X_test_round)
    pred_round = clf_round.predict(X_test_round)
    
    acc_round = accuracy_score(y_test_round, pred_round)
    print(f"Round Model Accuracy: {acc_round:.2%}")
    
    joblib.dump(clf_round, 'model_round.pkl')
    
    print("\nAll models trained and saved.")

if __name__ == "__main__":
    train_props()
