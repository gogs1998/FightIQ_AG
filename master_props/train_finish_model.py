import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split

def train_finish_model():
    print("=== Prop Hunter: Training 'Finish' Model (GTD vs ITD) ===")
    
    # 1. Load Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    # Load Lean Features
    with open(f'{BASE_DIR}/prop_hunter/features.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    print(f"Using {len(features)} Lean Features.")
    
    # 2. Prepare Target (Finish vs Decision)
    # result column contains: 'Decision - Unanimous', 'KO/TKO', 'Submission', etc.
    # We need to create a binary target: 1 = Finish, 0 = Decision
    
    def is_finish(method):
        if pd.isna(method): return 0
        m = str(method).lower()
        if 'decision' in m: return 0
        return 1
        
    df['is_finish'] = df['result'].apply(is_finish)
    
    # Filter valid odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['is_finish'].values
    
    print(f"Total Samples: {len(df)}")
    print(f"Finish Rate: {y.mean():.2%}")
    
    # 3. Split (Time Based)
    split_date = '2024-01-01'
    mask_train = df['event_date'] < split_date
    mask_test = df['event_date'] >= split_date
    
    X_train = X[mask_train]
    X_test = X[mask_test]
    y_train = y[mask_train]
    y_test = y[mask_test]
    
    print(f"Train Set: {len(X_train)}")
    print(f"Test Set:  {len(X_test)}")
    
    # 4. Train XGBoost
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
        
    print("Using Optimized Boruta Params...")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print(f"\n=== 'Finish' Model Results (2024-2025) ===")
    print(f"Accuracy: {acc:.4%}")
    print(f"Log Loss: {ll:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Predicted Finishes: {preds.sum()} / {len(preds)}")
    print(f"Actual Finishes:    {y_test.sum()} / {len(y_test)}")
    
    # Save Model
    joblib.dump(model, f'{BASE_DIR}/prop_hunter/model_finish.pkl')
    print("\nSaved model to prop_hunter/model_finish.pkl")

if __name__ == "__main__":
    train_finish_model()
