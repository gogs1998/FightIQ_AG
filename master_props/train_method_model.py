import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split

def train_method_model():
    print("=== Prop Hunter: Training 'Method' Model (KO vs Sub) ===")
    
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
    
    # 2. Prepare Target (KO vs Sub)
    # Filter ONLY fights that finished
    def get_method_target(method):
        if pd.isna(method): return None
        m = str(method).lower()
        if 'decision' in m: return None # Ignore decisions
        if 'ko' in m or 'tko' in m: return 1 # KO class
        if 'submission' in m: return 0 # Sub class
        return None # DQ/Draw/etc
        
    df['method_target'] = df['result'].apply(get_method_target)
    
    # Drop rows where target is None (Decisions or weird results)
    df_clean = df.dropna(subset=['method_target']).copy()
    
    # Filter valid odds
    has_odds = (df_clean['f_1_odds'].notna()) & (df_clean['f_1_odds'] > 1.0) & \
               (df_clean['f_2_odds'].notna()) & (df_clean['f_2_odds'] > 1.0)
    df_clean = df_clean[has_odds].copy()
    df_clean['event_date'] = pd.to_datetime(df_clean['event_date'])
    df_clean = df_clean.sort_values('event_date')
    
    X = df_clean[[c for c in features if c in df_clean.columns]].fillna(0)
    y = df_clean['method_target'].values
    
    print(f"Total Finish Samples: {len(df_clean)}")
    print(f"KO Rate (in finishes): {y.mean():.2%}")
    
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
    
    print(f"\n=== 'Method' Model Results (2024-2025) ===")
    print(f"Accuracy (KO vs Sub): {acc:.4%}")
    print(f"Log Loss: {ll:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Predicted KOs: {preds.sum()} / {len(preds)}")
    print(f"Actual KOs:    {y_test.sum()} / {len(y_test)}")
    
    # Save Model
    joblib.dump(model, f'{BASE_DIR}/prop_hunter/model_method.pkl')
    print("\nSaved model to prop_hunter/model_method.pkl")

if __name__ == "__main__":
    train_method_model()
