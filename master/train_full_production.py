import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict, KFold

# Import shared architecture
from models import SiameseMatchupNet, symmetric_loss, prepare_siamese_data

def train_full_production():
    print("=== Training FULL Production Models (2010-2025) ===")
    print("Objective: Maximize predictive power by training on ALL available data.")
    
    # 1. Load Data
    print("Loading full dataset...")
    df = pd.read_csv('data/training_data.csv')
    
    # Load Features & Params
    with open('features.json', 'r') as f: features = json.load(f)
    with open('params.json', 'r') as f: params = json.load(f)['best_params']
    
    # Filter valid odds
    if 'f_1_odds' in features and 'f_2_odds' in features:
        has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
                   (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
        df = df[has_odds].copy()
        
    X_df = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    print(f"Training on {len(df)} fights (Full History).")
    
    # 2. Train XGBoost (Full)
    print("Training XGBoost (Production)...")
    xgb_model = xgb.XGBClassifier(
        max_depth=params['xgb_max_depth'],
        learning_rate=params['xgb_learning_rate'],
        n_estimators=params['xgb_n_estimators'],
        min_child_weight=params['xgb_min_child_weight'],
        subsample=params['xgb_subsample'],
        colsample_bytree=params['xgb_colsample_bytree'],
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_df, y)
    
    # 3. Train Siamese (Full)
    print("Training Siamese Network (Production)...")
    f1, f2, input_dim, siamese_cols = prepare_siamese_data(X_df, features)
    
    scaler = StandardScaler()
    combined = np.concatenate([f1, f2], axis=0)
    scaler.fit(combined)
    
    f1_scaled = scaler.transform(f1)
    f2_scaled = scaler.transform(f2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
    
    train_ds = TensorDataset(
        torch.FloatTensor(f1_scaled),
        torch.FloatTensor(f2_scaled),
        torch.FloatTensor(y)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    optimizer = optim.Adam(siamese_model.parameters(), lr=params['siamese_lr'])
    
    for epoch in range(params['siamese_epochs']):
        siamese_model.train()
        for b_f1, b_f2, b_y in train_loader:
            b_f1, b_f2, b_y = b_f1.to(device), b_f2.to(device), b_y.to(device)
            optimizer.zero_grad()
            loss = symmetric_loss(siamese_model, b_f1, b_f2, b_y)
            loss.backward()
            optimizer.step()
            
    # 4. Train Calibrators (via CV)
    print("Training Isotonic Calibrators (via 5-Fold CV)...")
    # We need unbiased predictions for the training set to fit the calibrator
    
    # XGB CV Probs
    xgb_cv_probs = cross_val_predict(xgb_model, X_df, y, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
    
    # Siamese CV Probs (Manual CV loop needed for PyTorch)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    siam_cv_probs = np.zeros(len(y))
    
    # We need to re-train Siamese 5 times for CV... this is expensive but necessary for proper calibration on full data.
    # Alternatively, we can just use the OOB predictions if we had a Random Forest, but for NN we need CV.
    # To save time, we'll do a quick CV training loop.
    
    print("  - Running Siamese CV...")
    for train_idx, val_idx in kf.split(X_df):
        # Subset
        f1_tr, f2_tr = f1_scaled[train_idx], f2_scaled[train_idx]
        y_tr = y[train_idx]
        f1_val, f2_val = f1_scaled[val_idx], f2_scaled[val_idx]
        
        # Train Temp Model
        temp_model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
        temp_opt = optim.Adam(temp_model.parameters(), lr=params['siamese_lr'])
        temp_ds = TensorDataset(torch.FloatTensor(f1_tr), torch.FloatTensor(f2_tr), torch.FloatTensor(y_tr))
        temp_loader = DataLoader(temp_ds, batch_size=64, shuffle=True)
        
        for _ in range(params['siamese_epochs']): # Use same epochs
            temp_model.train()
            for b_f1, b_f2, b_y in temp_loader:
                b_f1, b_f2, b_y = b_f1.to(device), b_f2.to(device), b_y.to(device)
                temp_opt.zero_grad()
                loss = symmetric_loss(temp_model, b_f1, b_f2, b_y)
                loss.backward()
                temp_opt.step()
                
        # Predict
        temp_model.eval()
        with torch.no_grad():
            p = temp_model(torch.FloatTensor(f1_val).to(device), torch.FloatTensor(f2_val).to(device)).cpu().numpy()
        siam_cv_probs[val_idx] = p.flatten()

    # Fit Isotonic
    iso_xgb = IsotonicRegression(out_of_bounds='clip')
    iso_xgb.fit(xgb_cv_probs, y)
    
    iso_siam = IsotonicRegression(out_of_bounds='clip')
    iso_siam.fit(siam_cv_probs, y)
    
    # 5. Save Everything
    print("Saving Production Models...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(xgb_model, 'models/xgb_production.pkl')
    torch.save(siamese_model.state_dict(), 'models/siamese_production.pth')
    joblib.dump(scaler, 'models/siamese_scaler_production.pkl')
    joblib.dump(iso_xgb, 'models/iso_xgb_production.pkl')
    joblib.dump(iso_siam, 'models/iso_siam_production.pkl')
    
    with open('models/siamese_cols.json', 'w') as f:
        json.dump(siamese_cols, f) # This doesn't change
        
    print("✅ Full Production Models Saved.")

if __name__ == "__main__":
    train_full_production()
