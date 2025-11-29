import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def run_boruta_experiment():
    print("=== FightIQ Boruta Feature Selection Experiment ===")
    print("Objective: Identify the 'All-Star' features that drive the model's performance.")
    
    # 1. Load Data
    print("Loading data...")
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    with open(f'{BASE_DIR}/features.json', 'r') as f: features = json.load(f)
    
    # Filter valid odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    print(f"Running Boruta on {len(X)} samples with {len(features)} features...")
    print("This may take a few minutes...")
    
    # 2. Setup Boruta
    # Boruta works best with Random Forest, but can work with XGBoost.
    # We use RF here as it's the standard for Boruta.
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    
    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        verbose=2,
        random_state=42,
        max_iter=50 # Lower iter for speed in experiment
    )
    
    # 3. Fit
    boruta.fit(X.values, y)
    
    # 4. Results
    print("\n=== Boruta Results ===")
    
    confirmed = []
    tentative = []
    rejected = []
    
    for i, feat in enumerate(features):
        if boruta.support_[i]:
            confirmed.append(feat)
        elif boruta.support_weak_[i]:
            tentative.append(feat)
        else:
            rejected.append(feat)
            
    print(f"✅ Confirmed Important: {len(confirmed)}")
    print(f"⚠️ Tentative: {len(tentative)}")
    print(f"❌ Rejected: {len(rejected)}")
    
    print("\nTop 10 Confirmed Features:")
    for f in confirmed[:10]: print(f"  - {f}")
    
    # Save results
    results = {
        "confirmed": confirmed,
        "tentative": tentative,
        "rejected": rejected
    }
    
    with open(f'{BASE_DIR}/experiment_2/boruta_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved results to {BASE_DIR}/experiment_2/boruta_results.json")
    
    # 5. Quick Test: Train XGB on Confirmed Only
    print("\n--- Validation: XGBoost on Confirmed Features Only ---")
    X_conf = X[confirmed]
    X_train, X_test, y_train, y_test = train_test_split(X_conf, y, test_size=0.2, random_state=42)
    
    xgb_boruta = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_boruta.fit(X_train, y_train)
    acc = xgb_boruta.score(X_test, y_test)
    
    print(f"Accuracy with {len(confirmed)} features: {acc:.4%}")

if __name__ == "__main__":
    run_boruta_experiment()
