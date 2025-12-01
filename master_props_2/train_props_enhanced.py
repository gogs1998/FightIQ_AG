import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_enhanced_props():
    print("=== Master 3: Training Enhanced Prop Models ===")
    
    # 1. Load Data & Features
    print("Loading enhanced data...")
    # Path relative to master_props_2
    df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Load Features
    with open('../master_3/features_enhanced.json', 'r') as f:
        features = json.load(f)
        
    print(f"Total Fights: {len(df)}")
    print(f"Feature Count: {len(features)}")
    
    # 2. Prepare Targets
    # Finish: 1 if not Decision, 0 if Decision
    df['is_finish'] = df['result'].apply(lambda x: 0 if 'Decision' in str(x) else 1)
    
    # Method: 0=KO, 1=Sub (Only for finishes)
    # We need to be careful with "Draws" or "DQ". 
    # Usually "KO/TKO" or "Submission".
    def get_method_label(res):
        res = str(res).lower()
        if 'ko' in res or 'tko' in res: return 0
        if 'submission' in res: return 1
        return -1
    
    df['method_label'] = df['result'].apply(get_method_label)
    
    # Round: 0-4 (mapped from 1-5)
    # Only for finishes.
    def get_round_label(r):
        try:
            return int(r) - 1
        except:
            return -1
            
    df['round_label'] = df['finish_round'].apply(get_round_label)
    
    # 3. Split Data (Time-based)
    # Train on 2010-2023, Validate on 2024-2025 (just to check, but we save for prod)
    # Actually, for PRODUCTION models, we should train on EVERYTHING up to the current date?
    # Or stick to the 2023 cutoff to verify against our holdout?
    # Let's train on 2010-2023 to verify first.
    
    train_mask = df['event_date'] < '2024-01-01'
    test_mask = df['event_date'] >= '2024-01-01'
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    X_train = train_df[features].fillna(0)
    y_train_finish = train_df['is_finish']
    
    # Method Train Set (Finishes Only)
    mask_finish_train = (train_df['is_finish'] == 1) & (train_df['method_label'] != -1)
    X_train_method = train_df.loc[mask_finish_train, features].fillna(0)
    y_train_method = train_df.loc[mask_finish_train, 'method_label']
    
    # Round Train Set (Finishes Only)
    mask_round_train = (train_df['is_finish'] == 1) & (train_df['round_label'] != -1)
    X_train_round = train_df.loc[mask_round_train, features].fillna(0)
    y_train_round = train_df.loc[mask_round_train, 'round_label']
    
    # 4. Train Models
    print("\nTraining Finish Model...")
    model_finish = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model_finish.fit(X_train, y_train_finish)
    
    print("Training Method Model (KO vs Sub)...")
    model_method = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model_method.fit(X_train_method, y_train_method)
    
    print("Training Round Model (1-5)...")
    model_round = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=5,
        random_state=42,
        n_jobs=-1
    )
    model_round.fit(X_train_round, y_train_round)
    
    # 5. Save Models
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_finish, 'models/prop_enhanced_finish.pkl')
    joblib.dump(model_method, 'models/prop_enhanced_method.pkl')
    joblib.dump(model_round, 'models/prop_enhanced_round.pkl')
    
    print("\nModels saved to master_props_2/models/prop_enhanced_*.pkl")
    
    # 6. Quick Validation
    print("\n=== Validation (2024-2025) ===")
    X_test = test_df[features].fillna(0)
    
    # Finish Acc
    p_finish = model_finish.predict(X_test)
    acc_finish = accuracy_score(test_df['is_finish'], p_finish)
    print(f"Finish Accuracy: {acc_finish:.2%}")
    
    # Method Acc (on actual finishes)
    mask_finish_test = (test_df['is_finish'] == 1) & (test_df['method_label'] != -1)
    if mask_finish_test.sum() > 0:
        p_method = model_method.predict(test_df.loc[mask_finish_test, features].fillna(0))
        acc_method = accuracy_score(test_df.loc[mask_finish_test, 'method_label'], p_method)
        print(f"Method Accuracy (on finishes): {acc_method:.2%}")
        
    # Round Acc (on actual finishes)
    mask_round_test = (test_df['is_finish'] == 1) & (test_df['round_label'] != -1)
    if mask_round_test.sum() > 0:
        p_round = model_round.predict(test_df.loc[mask_round_test, features].fillna(0))
        acc_round = accuracy_score(test_df.loc[mask_round_test, 'round_label'], p_round)
        print(f"Round Accuracy (on finishes): {acc_round:.2%}")

if __name__ == "__main__":
    train_enhanced_props()
