import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score, log_loss

# --- Siamese Architecture ---
class SiameseMatchupNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(SiameseMatchupNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.comparator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward_one(self, x): return self.encoder(x)
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.comparator(combined)

def symmetric_loss(model, x1, x2, y):
    criterion = nn.BCELoss()
    pred_1 = model(x1, x2).squeeze()
    pred_1 = torch.clamp(pred_1, 1e-7, 1 - 1e-7)
    loss_1 = criterion(pred_1, y)
    
    pred_2 = model(x2, x1).squeeze()
    pred_2 = torch.clamp(pred_2, 1e-7, 1 - 1e-7)
    loss_2 = criterion(pred_2, 1.0 - y)
    return 0.5 * (loss_1 + loss_2)

def prepare_siamese_data(df, features):
    f1_cols = [c for c in features if c.startswith('f_1_')]
    f2_cols = [c for c in features if c.startswith('f_2_')]
    valid_pairs = []
    for c1 in f1_cols:
        c2 = c1.replace('f_1_', 'f_2_')
        if c2 in f2_cols: valid_pairs.append((c1, c2))
    f1_data = df[[p[0] for p in valid_pairs]].values
    f2_data = df[[p[1] for p in valid_pairs]].values
    return f1_data, f2_data, len(valid_pairs), valid_pairs

# --- Main Script ---
def train_gold_holdout():
    print("=== FightIQ: Gold Standard HOLDOUT Test (Train < 2024, Test >= 2024) ===")
    
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    
    # 1. Load Data
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    with open(f'{BASE_DIR}/experiment_2/boruta_results.json', 'r') as f:
        features = json.load(f)['confirmed']
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
        
    # Filter valid odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # SPLIT
    split_date = '2024-01-01'
    mask_train = df['event_date'] < split_date
    mask_test = df['event_date'] >= split_date
    
    train_df = df[mask_train].copy()
    test_df = df[mask_test].copy()
    
    print(f"Training on {len(train_df)} fights (2010-2023).")
    print(f"Testing on {len(test_df)} fights (2024-2025).")
    
    X_train = train_df[[c for c in features if c in df.columns]].fillna(0)
    y_train = train_df['target'].values
    X_test = test_df[[c for c in features if c in df.columns]].fillna(0)
    y_test = test_df['target'].values
    
    # 2. Train XGBoost
    print("\n[1/3] Training XGBoost...")
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    
    # 3. Train Siamese
    print("\n[2/3] Training Siamese...")
    f1_tr, f2_tr, input_dim, _ = prepare_siamese_data(train_df, features)
    f1_te, f2_te, _, _ = prepare_siamese_data(test_df, features)
    
    if np.isnan(f1_tr).any() or np.isnan(f2_tr).any():
        print("WARNING: NaNs found in training data! Filling with 0.")
        f1_tr = np.nan_to_num(f1_tr)
        f2_tr = np.nan_to_num(f2_tr)
        
    scaler = StandardScaler()
    combined = np.concatenate([f1_tr, f2_tr], axis=0)
    scaler.fit(combined)
    
    f1_tr_s = scaler.transform(f1_tr)
    f2_tr_s = scaler.transform(f2_tr)
    f1_te_s = scaler.transform(f1_te)
    f2_te_s = scaler.transform(f2_te)
    
    if np.isnan(f1_tr_s).any():
        f1_tr_s = np.nan_to_num(f1_tr_s)
        f2_tr_s = np.nan_to_num(f2_tr_s)
        f1_te_s = np.nan_to_num(f1_te_s)
        f2_te_s = np.nan_to_num(f2_te_s)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim).to(device)
    
    # Added drop_last=True
    train_ds = TensorDataset(torch.FloatTensor(f1_tr_s), torch.FloatTensor(f2_tr_s), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    optimizer = optim.Adam(siamese_model.parameters(), lr=0.001)
    
    for epoch in range(15):
        siamese_model.train()
        for b_f1, b_f2, b_y in train_loader:
            b_f1, b_f2, b_y = b_f1.to(device), b_f2.to(device), b_y.to(device)
            optimizer.zero_grad()
            loss = symmetric_loss(siamese_model, b_f1, b_f2, b_y)
            loss.backward()
            optimizer.step()
            
    # 4. Train Calibrators (CV on Train Set)
    print("\n[3/3] Training Calibrators...")
    xgb_cv_probs = cross_val_predict(xgb_model, X_train, y_train, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
    
    # Siamese CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    siam_cv_probs = np.zeros(len(y_train))
    
    print("  Running Siamese CV...")
    for tr_idx, val_idx in kf.split(X_train):
        f1_sub, f2_sub = f1_tr_s[tr_idx], f2_tr_s[tr_idx]
        y_sub = y_train[tr_idx]
        f1_val, f2_val = f1_tr_s[val_idx], f2_tr_s[val_idx]
        
        tmp = SiameseMatchupNet(input_dim).to(device)
        opt = optim.Adam(tmp.parameters(), lr=0.001)
        ds = TensorDataset(torch.FloatTensor(f1_sub), torch.FloatTensor(f2_sub), torch.FloatTensor(y_sub))
        # Added drop_last=True here too
        dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
        
        for _ in range(10):
            tmp.train()
            for b1, b2, by in dl:
                b1, b2, by = b1.to(device), b2.to(device), by.to(device)
                opt.zero_grad()
                loss = symmetric_loss(tmp, b1, b2, by)
                loss.backward()
                opt.step()
        
        tmp.eval()
        with torch.no_grad():
            p = tmp(torch.FloatTensor(f1_val).to(device), torch.FloatTensor(f2_val).to(device)).cpu().numpy()
        siam_cv_probs[val_idx] = p.flatten()
        
    iso_xgb = IsotonicRegression(out_of_bounds='clip').fit(xgb_cv_probs, y_train)
    iso_siam = IsotonicRegression(out_of_bounds='clip').fit(siam_cv_probs, y_train)
    
    # 5. Evaluate on Holdout (2024-2025)
    print("\n=== EVALUATION (2024-2025) ===")
    
    # Raw Probs
    p_xgb_raw = xgb_model.predict_proba(X_test)[:, 1]
    
    siamese_model.eval()
    with torch.no_grad():
        p_siam_raw = siamese_model(torch.FloatTensor(f1_te_s).to(device), torch.FloatTensor(f2_te_s).to(device)).cpu().numpy().flatten()
        
    # Calibrated Probs
    p_xgb_cal = iso_xgb.transform(p_xgb_raw)
    p_siam_cal = iso_siam.transform(p_siam_raw)
    
    # Ensemble (0.6 XGB / 0.4 Siam)
    p_ens = 0.6 * p_xgb_cal + 0.4 * p_siam_cal
    
    preds = (p_ens > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, p_ens)
    
    print(f"Accuracy: {acc:.2%}")
    print(f"Log Loss: {ll:.4f}")
    
    # ROI Check
    profit = 0
    invested = 0
    for i, idx in enumerate(test_df.index):
        row = test_df.loc[idx]
        prob = p_ens[i]
        target = y_test[i]
        odds_1, odds_2 = row['f_1_odds'], row['f_2_odds']
        
        if prob > 0.5:
            implied = 1/odds_1
            if (prob - implied) > 0.05:
                if target == 1: profit += (odds_1 - 1)
                else: profit -= 1
                invested += 1
        else:
            implied = 1/odds_2
            if ((1-prob) - implied) > 0.05:
                if target == 0: profit += (odds_2 - 1)
                else: profit -= 1
                invested += 1
                
    roi = profit / invested if invested > 0 else 0
    print(f"Value Sniper ROI: {roi:.1%} ({invested} bets)")

if __name__ == "__main__":
    train_gold_holdout()
