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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# --- Architecture ---
class SiameseMatchupNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64): # Optimized hidden_dim
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, f1, f2):
        e1 = self.encoder(f1)
        e2 = self.encoder(f2)
        combined = torch.cat([e1, e2], dim=1)
        return self.classifier(combined).squeeze()

def symmetric_loss(model, f1, f2, y):
    # Forward pass: (f1, f2) -> y
    pred1 = model(f1, f2)
    loss1 = nn.BCELoss()(pred1, y)
    
    # Symmetry constraint: (f2, f1) -> 1-y
    pred2 = model(f2, f1)
    loss2 = nn.BCELoss()(pred2, 1.0 - y)
    
    return 0.5 * (loss1 + loss2)

# --- Data Prep ---
def prepare_siamese_data(X_df, features):
    """
    Prepare data for Siamese network using robust pair finding.
    """
    pairs = set()
    all_cols = set(X_df.columns)
    
    for feat in features:
        base = None
        f1_col = None
        f2_col = None
        
        if feat.startswith('diff_'):
            base = feat[5:] 
            if f"f_1_{base}" in all_cols and f"f_2_{base}" in all_cols:
                f1_col = f"f_1_{base}"
                f2_col = f"f_2_{base}"
            elif f"{base}_f_1" in all_cols and f"{base}_f_2" in all_cols:
                f1_col = f"{base}_f_1"
                f2_col = f"{base}_f_2"
                
        elif '_f_1' in feat or 'f_1_' in feat:
            if feat.startswith('f_1_'):
                f1_col = feat
                f2_col = feat.replace('f_1_', 'f_2_')
            else:
                f1_col = feat
                f2_col = feat.replace('_f_1', '_f_2')
            if f2_col not in all_cols: f1_col = None
            
        elif '_f_2' in feat or 'f_2_' in feat:
            if feat.startswith('f_2_'):
                f2_col = feat
                f1_col = feat.replace('f_2_', 'f_1_')
            else:
                f2_col = feat
                f1_col = feat.replace('_f_2', '_f_1')
            if f1_col not in all_cols: f1_col = None
            
        if f1_col and f2_col:
            if (f1_col, f2_col) not in pairs:
                pairs.add((f1_col, f2_col))

    pairs = list(pairs)
    
    numeric_pairs = []
    for c1, c2 in pairs:
        if c1 in X_df.columns and c2 in X_df.columns:
            numeric_pairs.append((c1, c2))
            
    f1_feats = [p[0] for p in numeric_pairs]
    f2_feats = [p[1] for p in numeric_pairs]
    
    if not f1_feats:
        return np.zeros((len(X_df), 1)), np.zeros((len(X_df), 1)), 1, []

    f1_data = X_df[f1_feats].values
    f2_data = X_df[f2_feats].values
    
    return f1_data, f2_data, len(f1_feats), f1_feats

def train_final_model():
    print("Loading data...")
    df = pd.read_csv('v2/data/training_data_v2.csv')
    
    # Load safe features
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    print(f"Using {len(features)} features.")
    
    # Prepare Data
    X_df = df[[c for c in features if c in df.columns]]
    X_df = X_df.fillna(0)
    y = df['target'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 1. Train XGBoost (Optimized) ---
    print("Training Optimized XGBoost...")
    # Best Params from Optuna
    xgb_params = {
        'max_depth': 9,
        'learning_rate': 0.03937396054033403,
        'n_estimators': 187,
        'min_child_weight': 9,
        'subsample': 0.7875123486508065,
        'colsample_bytree': 0.7725744179294693,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)
    
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_acc = accuracy_score(y_test, (xgb_probs > 0.5).astype(int))
    print(f"XGBoost Accuracy: {xgb_acc:.4f}")
    
    # --- 2. Train Siamese (Optimized) ---
    print("Training Optimized Siamese Network...")
    
    f1_train, f2_train, input_dim, siamese_cols = prepare_siamese_data(X_train, features)
    f1_test, f2_test, _, _ = prepare_siamese_data(X_test, features)
    
    # Save Siamese columns for inference
    with open('v2/models/siamese_cols.json', 'w') as f:
        json.dump(siamese_cols, f)
        
    # Scale Siamese Data
    scaler = StandardScaler()
    # Fit on combined train data to ensure consistent scaling
    combined_train = np.concatenate([f1_train, f2_train], axis=0)
    scaler.fit(combined_train)
    
    f1_train = scaler.transform(f1_train)
    f2_train = scaler.transform(f2_train)
    f1_test = scaler.transform(f1_test)
    f2_test = scaler.transform(f2_test)
    
    # Save Scaler
    joblib.dump(scaler, 'v2/models/siamese_scaler.pkl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Best Params from Optuna
    siamese_params = {
        'hidden_dim': 64,
        'epochs': 20,
        'lr': 0.0010106275348677788
    }
    
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=siamese_params['hidden_dim']).to(device)
    
    train_ds = TensorDataset(
        torch.FloatTensor(f1_train),
        torch.FloatTensor(f2_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(siamese_model.parameters(), lr=siamese_params['lr'])
    
    for epoch in range(siamese_params['epochs']):
        siamese_model.train()
        for b_f1, b_f2, b_y in train_loader:
            b_f1, b_f2, b_y = b_f1.to(device), b_f2.to(device), b_y.to(device)
            optimizer.zero_grad()
            loss = symmetric_loss(siamese_model, b_f1, b_f2, b_y)
            loss.backward()
            optimizer.step()
            
    siamese_model.eval()
    with torch.no_grad():
        t_f1 = torch.FloatTensor(f1_test).to(device)
        t_f2 = torch.FloatTensor(f2_test).to(device)
        siamese_probs = siamese_model(t_f1, t_f2).cpu().numpy()
        
    siamese_acc = accuracy_score(y_test, (siamese_probs > 0.5).astype(int))
    print(f"Siamese Accuracy: {siamese_acc:.4f}")
    
    # --- 3. Ensemble ---
    xgb_weight = 0.40524147388041054 # From Optuna
    
    ens_probs = xgb_weight * xgb_probs + (1 - xgb_weight) * siamese_probs
    ens_preds = (ens_probs > 0.5).astype(int)
    
    final_acc = accuracy_score(y_test, ens_preds)
    final_ll = log_loss(y_test, ens_probs)
    
    print(f"\n=== FINAL OPTIMIZED MODEL ===")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Log Loss: {final_ll:.4f}")
    
    # Save Models
    print("Saving models...")
    os.makedirs('v2/models', exist_ok=True)
    
    joblib.dump(xgb_model, 'v2/models/xgb_optimized.pkl')
    torch.save(siamese_model.state_dict(), 'v2/models/siamese_optimized.pth')
    
    # Save metadata
    metadata = {
        'xgb_weight': xgb_weight,
        'features': features,
        'accuracy': final_acc,
        'log_loss': final_ll
    }
    with open('v2/models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Done. Models saved to v2/models/")

if __name__ == "__main__":
    train_final_model()
