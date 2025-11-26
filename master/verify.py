import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# Import shared architecture
from models import SiameseMatchupNet, symmetric_loss, prepare_siamese_data

def run_verification():
    print("=== Starting Reproducible Verification (5-Fold CV) ===")
    
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists('data/training_data.csv'):
        print("Error: data/training_data.csv not found.")
        return

    df = pd.read_csv('data/training_data.csv')
    
    with open('features.json', 'r') as f:
        features = json.load(f)
    print(f"Using {len(features)} features.")
    
    with open('params.json', 'r') as f:
        best = json.load(f)
        params = best['best_params']
    print("Loaded hyperparameters.")
    
    X_df = df[[c for c in features if c in df.columns]]
    X_df = X_df.fillna(0)
    y = df['target'].values
    
    # Sort by date for TimeSeriesSplit
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X_df = df[[c for c in features if c in df.columns]]
    X_df = X_df.fillna(0)
    y = df['target'].values
    
    # Time Series Split (5 Splits)
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    log_losses = []
    
    fold = 1
    for train_idx, test_idx in tscv.split(X_df, y):
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
        f1_train, f2_train, input_dim, _ = prepare_siamese_data(X_train, features)
        f1_test, f2_test, _, _ = prepare_siamese_data(X_test, features)
        
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
        
    print("\n=== Verification Results (5-Fold CV) ===")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Mean Log Loss: {np.mean(log_losses):.4f}")
    print(f"Individual Folds: {[round(x, 4) for x in accuracies]}")

if __name__ == "__main__":
    run_verification()
