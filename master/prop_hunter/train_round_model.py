import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split

def train_round_model():
    print("=== Prop Hunter: Training 'Round' Model (1-5) ===")
    
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
    
    # 2. Prepare Target (Round Number)
    # Filter ONLY fights that finished (Decision rounds are not 'finishes')
    def is_finish(method):
        if pd.isna(method): return False
        m = str(method).lower()
        return 'decision' not in m
        
    df['is_finish'] = df['result'].apply(is_finish)
    df_clean = df[df['is_finish']].copy()
    
    # Target: finish_round - 1 (to make it 0-indexed for XGBoost: 0=R1, 1=R2...)
    df_clean['round_target'] = df_clean['finish_round'].astype(int) - 1
    
    # Filter valid odds
    has_odds = (df_clean['f_1_odds'].notna()) & (df_clean['f_1_odds'] > 1.0) & \
               (df_clean['f_2_odds'].notna()) & (df_clean['f_2_odds'] > 1.0)
    df_clean = df_clean[has_odds].copy()
    df_clean['event_date'] = pd.to_datetime(df_clean['event_date'])
    df_clean = df_clean.sort_values('event_date')
    
    X = df_clean[[c for c in features if c in df_clean.columns]].fillna(0)
    y = df_clean['round_target'].values
    
    print(f"Total Finish Samples: {len(df_clean)}")
    print(f"Class Distribution: {np.bincount(y)}")
    
    # 3. Split (Time Based)
    split_date = '2024-01-01'
    mask_train = df_clean['event_date'] < split_date
    mask_test = df_clean['event_date'] >= split_date
    
    X_train = X[mask_train]
    X_test = X[mask_test]
    y_train = y[mask_train]
    y_test = y[mask_test]
    
    print(f"Train Set: {len(X_train)}")
    print(f"Test Set:  {len(X_test)}")
    
    # 4. Train XGBoost (Multi-Class)
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
        
    # Adjust params for multi-class
    params['objective'] = 'multi:softprob'
    params['num_class'] = 5 # R1, R2, R3, R4, R5
    params['eval_metric'] = 'mlogloss'
    
    print("Using Optimized Boruta Params (Adapted for Multi-Class)...")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print(f"\n=== 'Round' Model Results (2024-2025) ===")
    print(f"Accuracy (Exact Round): {acc:.4%}")
    print(f"Log Loss: {ll:.4f}")
    print("Confusion Matrix (Rows=Actual, Cols=Pred):")
    print(cm)
    
    # Save Model
    joblib.dump(model, f'{BASE_DIR}/prop_hunter/model_round.pkl')
    print("\nSaved model to prop_hunter/model_round.pkl")

if __name__ == "__main__":
    train_round_model()
