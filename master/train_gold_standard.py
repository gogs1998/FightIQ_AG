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

# --- Siamese Architecture (Inline for completeness) ---
class SiameseMatchupNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super(SiameseMatchupNet, self).__init__()
        
        # Shared Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Comparator
        self.comparator = nn.Sequential(
            nn.Linear(64, 32), # 32 + 32 concatenated
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward_one(self, x):
        return self.encoder(x)
        
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.comparator(combined)

def symmetric_loss(model, x1, x2, y):
    # Loss = 0.5 * (BCELoss(A,B, y) + BCELoss(B,A, 1-y))
    criterion = nn.BCELoss()
    
    pred_1 = model(x1, x2).squeeze()
    loss_1 = criterion(pred_1, y)
    
    pred_2 = model(x2, x1).squeeze()
    loss_2 = criterion(pred_2, 1.0 - y)
    
    return 0.5 * (loss_1 + loss_2)

def prepare_siamese_data(df, features):
    # Split features into f_1 and f_2
    f1_cols = [c for c in features if c.startswith('f_1_')]
    f2_cols = [c for c in features if c.startswith('f_2_')]
    
    # Ensure matching pairs
    valid_pairs = []
    for c1 in f1_cols:
        c2 = c1.replace('f_1_', 'f_2_')
        if c2 in f2_cols:
            valid_pairs.append((c1, c2))
            
    f1_data = df[[p[0] for p in valid_pairs]].values
    f2_data = df[[p[1] for p in valid_pairs]].values
    
    return f1_data, f2_data, len(valid_pairs), valid_pairs

# --- Main Training Script ---
def train_gold_standard():
    print("=== FightIQ: Training GOLD STANDARD Ensemble (2010-2025) ===")
    
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    MODELS_DIR = f'{BASE_DIR}/models'
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("Loading Dataset...")
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    # 2. Load Features & Params
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
    
    X_df = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    print(f"Training on {len(df)} fights.")
    
    # 3. Train XGBoost (Optimized)
    print("\n[1/3] Training XGBoost (Optimized)...")
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(X_df, y)
    
    # 4. Train Siamese Network
    print("\n[2/3] Training Siamese Network (PyTorch)...")
    f1, f2, input_dim, siamese_pairs = prepare_siamese_data(X_df, features)
    
    scaler = StandardScaler()
    combined = np.concatenate([f1, f2], axis=0)
    scaler.fit(combined)
    
    f1_scaled = scaler.transform(f1)
    f2_scaled = scaler.transform(f2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    siamese_model = SiameseMatchupNet(input_dim).to(device)
    
    train_ds = TensorDataset(
        torch.FloatTensor(f1_scaled),
        torch.FloatTensor(f2_scaled),
        torch.FloatTensor(y)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    optimizer = optim.Adam(siamese_model.parameters(), lr=0.001)
    
    # Train Loop
    epochs = 15
    for epoch in range(epochs):
        siamese_model.train()
        total_loss = 0
        for b_f1, b_f2, b_y in train_loader:
            b_f1, b_f2, b_y = b_f1.to(device), b_f2.to(device), b_y.to(device)
            optimizer.zero_grad()
            loss = symmetric_loss(siamese_model, b_f1, b_f2, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    # 5. Train Isotonic Calibrators (5-Fold CV)
    print("\n[3/3] Training Isotonic Calibrators (CV)...")
    
    # XGB CV Probs
    xgb_cv_probs = cross_val_predict(xgb_model, X_df, y, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
    
    # Siamese CV Probs
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    siam_cv_probs = np.zeros(len(y))
    
    print("  Running Siamese CV (this takes a moment)...")
    for train_idx, val_idx in kf.split(X_df):
        f1_tr, f2_tr = f1_scaled[train_idx], f2_scaled[train_idx]
        y_tr = y[train_idx]
        f1_val, f2_val = f1_scaled[val_idx], f2_scaled[val_idx]
        
        temp_model = SiameseMatchupNet(input_dim).to(device)
        temp_opt = optim.Adam(temp_model.parameters(), lr=0.001)
        temp_ds = TensorDataset(torch.FloatTensor(f1_tr), torch.FloatTensor(f2_tr), torch.FloatTensor(y_tr))
        temp_loader = DataLoader(temp_ds, batch_size=64, shuffle=True)
        
        for _ in range(10): # Reduced epochs for CV speed
            temp_model.train()
            for b_f1, b_f2, b_y in temp_loader:
                b_f1, b_f2, b_y = b_f1.to(device), b_f2.to(device), b_y.to(device)
                temp_opt.zero_grad()
                loss = symmetric_loss(temp_model, b_f1, b_f2, b_y)
                loss.backward()
                temp_opt.step()
                
        temp_model.eval()
        with torch.no_grad():
            p = temp_model(torch.FloatTensor(f1_val).to(device), torch.FloatTensor(f2_val).to(device)).cpu().numpy()
        siam_cv_probs[val_idx] = p.flatten()
        
    # Fit Calibrators
    iso_xgb = IsotonicRegression(out_of_bounds='clip')
    iso_xgb.fit(xgb_cv_probs, y)
    
    iso_siam = IsotonicRegression(out_of_bounds='clip')
    iso_siam.fit(siam_cv_probs, y)
    
    # 6. Save Everything
    print("\nSaving Gold Standard Models...")
    joblib.dump(xgb_model, f'{MODELS_DIR}/xgb_production.pkl')
    torch.save(siamese_model.state_dict(), f'{MODELS_DIR}/siamese_production.pth')
    joblib.dump(scaler, f'{MODELS_DIR}/siamese_scaler_production.pkl')
    joblib.dump(iso_xgb, f'{MODELS_DIR}/iso_xgb_production.pkl')
    joblib.dump(iso_siam, f'{MODELS_DIR}/iso_siam_production.pkl')
    
    with open(f'{MODELS_DIR}/siamese_cols.json', 'w') as f:
        json.dump([p[0] for p in siamese_pairs], f)
        
    print("✅ Gold Standard Training Complete.")

if __name__ == "__main__":
    train_gold_standard()
