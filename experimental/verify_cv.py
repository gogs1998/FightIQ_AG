import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# --- Architecture ---
class SiameseMatchupNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
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
    pred1 = model(f1, f2)
    loss1 = nn.BCELoss()(pred1, y)
    pred2 = model(f2, f1)
    loss2 = nn.BCELoss()(pred2, 1.0 - y)
    return 0.5 * (loss1 + loss2)

# --- Data Prep ---
def prepare_siamese_data(X_df, features):
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
        return np.zeros((len(X_df), 1)), np.zeros((len(X_df), 1)), 1

    f1_data = X_df[f1_feats].values
    f2_data = X_df[f2_feats].values
    
    return f1_data, f2_data, len(f1_feats)

def run_cv():
    print("Loading data...")
    df = pd.read_csv('v2/data/training_data_v2.csv')
    
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    print(f"Using {len(features)} features.")
    
    # Load Best Params
    with open('experimental/optuna_best_params.json', 'r') as f:
        best = json.load(f)
        params = best['best_params']
        
    print("Loaded Best Params:")
    print(json.dumps(params, indent=2))
    
    X_df = df[[c for c in features if c in df.columns]]
    X_df = X_df.fillna(0)
    y = df['target'].values
    
    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracies = []
    log_losses = []
    
    fold = 1
    for train_idx, test_idx in skf.split(X_df, y):
        print(f"\n--- Fold {fold}/5 ---")
        
        X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 1. XGBoost
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
        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        
        # 2. Siamese
        f1_train, f2_train, input_dim = prepare_siamese_data(X_train, features)
        f1_test, f2_test, _ = prepare_siamese_data(X_test, features)
        
        # Scale
        scaler = StandardScaler()
        combined_train = np.concatenate([f1_train, f2_train], axis=0)
        scaler.fit(combined_train)
        
        f1_train = scaler.transform(f1_train)
        f2_train = scaler.transform(f2_train)
        f1_test = scaler.transform(f1_test)
        f2_test = scaler.transform(f2_test)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        siamese_model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
        
        train_ds = TensorDataset(
            torch.FloatTensor(f1_train),
            torch.FloatTensor(f2_train),
            torch.FloatTensor(y_train)
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
                
        siamese_model.eval()
        with torch.no_grad():
            t_f1 = torch.FloatTensor(f1_test).to(device)
            t_f2 = torch.FloatTensor(f2_test).to(device)
            siamese_probs = siamese_model(t_f1, t_f2).cpu().numpy()
            
        # 3. Ensemble
        w = params['ensemble_xgb_weight']
        ens_probs = w * xgb_probs + (1 - w) * siamese_probs
        ens_preds = (ens_probs > 0.5).astype(int)
        
        acc = accuracy_score(y_test, ens_preds)
        ll = log_loss(y_test, ens_probs)
        
        print(f"Fold {fold} Accuracy: {acc:.4f}, Log Loss: {ll:.4f}")
        accuracies.append(acc)
        log_losses.append(ll)
        
        fold += 1
        
    print("\n=== Cross-Validation Results ===")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Mean Log Loss: {np.mean(log_losses):.4f}")
    print(f"Individual Folds: {[round(x, 4) for x in accuracies]}")

if __name__ == "__main__":
    run_cv()
