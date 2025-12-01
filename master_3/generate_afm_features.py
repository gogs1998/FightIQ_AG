import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- Configuration ---
BASE_DIR = 'd:/AntiGravity/FightIQ/master_2'

def generate_afm():
    print("=== Generating AFM Features (Original Method: LogReg + Noise) ===")
    
    # 1. Load Data
    df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    with open(f'{BASE_DIR}/features.json', 'r') as f: features = json.load(f)
    
    # 2. Prepare Data
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train Surrogate Model (Logistic Regression)
    # Train on pre-2024 data to avoid leakage, or full data?
    # Original script trained on pre-2024. We should probably train on full data 
    # but use CV predictions to be safe? 
    # Actually, to match the original exactly, let's train on pre-2024 for the surrogate
    # BUT, that leaves 2024-2025 un-surrogated? No, the original script trained on pre-2024
    # and then predicted on everything. Let's do that.
    
    print("Training Surrogate Model (Logistic Regression)...")
    mask_train = df['event_date'] < '2024-01-01'
    X_train_surr = X_scaled[mask_train]
    y_train_surr = y[mask_train]
    
    surrogate = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    surrogate.fit(X_train_surr, y_train_surr)
    
    # 4. Compute AFM Features (Perturbation)
    print("Computing AFM Features (Perturbation)...")
    
    n_perturbations = 50
    base_probs = surrogate.predict_proba(X_scaled)[:, 1]
    
    min_probs = base_probs.copy()
    max_probs = base_probs.copy()
    
    print(f"Simulating {n_perturbations} perturbations per fight...")
    
    # Vectorized Perturbation
    for i in tqdm(range(n_perturbations)):
        noise = np.random.normal(0, 0.2, X_scaled.shape) # 0.2 std dev noise
        X_noisy = X_scaled + noise
        probs_noisy = surrogate.predict_proba(X_noisy)[:, 1]
        
        min_probs = np.minimum(min_probs, probs_noisy)
        max_probs = np.maximum(max_probs, probs_noisy)
        
    # Calculate Metrics
    afm_upside = max_probs - base_probs
    afm_downside = base_probs - min_probs
    afm_skew = afm_upside - afm_downside
    
    afm_fragile = ((base_probs < 0.5) & (max_probs > 0.5)) | \
                  ((base_probs > 0.5) & (min_probs < 0.5))
    afm_fragile = afm_fragile.astype(int)
    
    # 5. Save Features
    df['afm_upside'] = afm_upside
    df['afm_downside'] = afm_downside
    df['afm_skew'] = afm_skew
    df['afm_fragile'] = afm_fragile
    
    # Save to new CSV
    df.to_csv(f'{BASE_DIR}/data/training_data_afm.csv', index=False)
    
    # Update features.json
    new_features = features + ['afm_upside', 'afm_downside', 'afm_skew', 'afm_fragile']
    with open(f'{BASE_DIR}/features_afm.json', 'w') as f:
        json.dump(new_features, f, indent=4)
        
    print("Saved AFM data to training_data_afm.csv and features_afm.json")

if __name__ == "__main__":
    generate_afm()
