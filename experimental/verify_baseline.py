import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

def verify_baseline():
    print("Loading data...")
    df = pd.read_csv('v2/data/training_data_v2.csv')
    
    # Load selected features (SAFE LIST)
    with open('features_elo.json', 'r') as f:
        selected_features = json.load(f)
        
    print(f"Using {len(selected_features)} features from features_elo.json")
    
    # Prepare X and y
    X_df = df[[c for c in selected_features if c in df.columns]]
    X_df = X_df.fillna(0)
    y = df['target'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training XGBoost on {X_train.shape[0]} samples...")
    
    # Simple XGBoost (similar to experimental setup)
    model = XGBClassifier(
        n_jobs=-1, 
        max_depth=5,
        n_estimators=100, 
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    print("\n=== Baseline Verification Results ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Log Loss: {ll:.4f}")
    
    if acc > 0.67:
        print("\nSUCCESS: Baseline accuracy replicated (>67%)")
    else:
        print("\nWARNING: Accuracy is lower than expected.")

if __name__ == "__main__":
    verify_baseline()
