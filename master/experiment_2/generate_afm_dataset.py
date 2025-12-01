import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os

def generate_and_save_afm():
    print("=== FightIQ: Generating AFM Features (Production) ===")
    
    # 1. Load Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    # 2. Load Base Features & Params
    with open(f'{BASE_DIR}/prop_hunter/features.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
        
    # 3. Train Surrogate Model (on full data for feature extraction)
    # Note: In strict backtesting, we should train only on past data.
    # But AFM is a "Sensitivity Analysis" of the model itself.
    # To avoid leakage, we will use 5-fold CV to generate AFM scores.
    
    from sklearn.model_selection import KFold
    
    df['afm_upside'] = 0.0
    df['afm_downside'] = 0.0
    df['afm_fragile'] = 0
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    X = df[features].fillna(0)
    y = df['target']
    
    print("Calculating AFM via 5-Fold CV...")
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        
        # Train Surrogate
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)
        
        # Generate Perturbations for Validation Set
        # 1. Base Prediction
        base_probs = clf.predict_proba(X_val)[:, 1]
        
        # 2. Perturb: Boost Fighter A (Upside)
        X_val_up = X_val.copy()
        if 'diff_elo' in X_val.columns: X_val_up['diff_elo'] += 50
        if 'diff_strike_acc' in X_val.columns: X_val_up['diff_strike_acc'] += 0.05
        probs_up = clf.predict_proba(X_val_up)[:, 1]
        
        # 3. Perturb: Nerf Fighter A (Downside)
        X_val_down = X_val.copy()
        if 'diff_elo' in X_val.columns: X_val_down['diff_elo'] -= 50
        if 'diff_strike_acc' in X_val.columns: X_val_down['diff_strike_acc'] -= 0.05
        probs_down = clf.predict_proba(X_val_down)[:, 1]
        
        # 4. Calculate AFM Metrics
        upside = probs_up - base_probs
        downside = base_probs - probs_down
        
        # Store
        df.iloc[val_idx, df.columns.get_loc('afm_upside')] = upside
        df.iloc[val_idx, df.columns.get_loc('afm_downside')] = downside
        
    # Fragility Flag (Top 20% Skew)
    df['afm_skew'] = df['afm_downside'] - df['afm_upside']
    threshold = df['afm_skew'].quantile(0.80)
    df['afm_fragile'] = (df['afm_skew'] > threshold).astype(int)
    
    print(f"Fragility Threshold: {threshold:.4f}")
    print(f"Fragile Fights identified: {df['afm_fragile'].sum()}")
    
    # Save
    save_path = f'{BASE_DIR}/data/training_data_with_afm.csv'
    df.to_csv(save_path, index=False)
    print(f"Saved dataset with AFM to: {save_path}")

if __name__ == "__main__":
    generate_and_save_afm()
