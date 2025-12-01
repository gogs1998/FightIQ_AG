import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import joblib
import optuna
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import Siamese
from models import SiameseMatchupNet, symmetric_loss, prepare_siamese_data

def run_optimization():
    print("=== Master 3 Optimization Phase ===")
    
    # 1. Load Data
    if not os.path.exists('data/training_data_enhanced.csv'):
        print("Error: data/training_data_enhanced.csv not found.")
        return
    df = pd.read_csv('data/training_data_enhanced.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Load Features
    with open('features_enhanced.json', 'r') as f:
        features = json.load(f)
        
    # Filter missing odds
    if 'f_1_odds' in features and 'f_2_odds' in features:
        has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
                   (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
        df = df[has_odds].copy()
        
    X_df = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    # Split (2024-01-01)
    split_date = '2024-01-01'
    mask_train = df['event_date'] < split_date
    mask_test = df['event_date'] >= split_date
    
    X_train = X_df[mask_train]
    y_train = y[mask_train]
    X_test = X_df[mask_test]
    y_test = y[mask_test]
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # --- STEP 1: BORUTA FEATURE SELECTION ---
    print("\n--- Step 1: Boruta Feature Selection ---")
    # We use a Random Forest for Boruta
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    
    # Boruta needs numpy arrays
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42, max_iter=50)
    feat_selector.fit(X_train.values, y_train)
    
    # Get selected features
    confirmed_mask = feat_selector.support_
    tentative_mask = feat_selector.support_weak_
    
    selected_features = [f for f, s in zip(features, confirmed_mask) if s]
    tentative_features = [f for f, s in zip(features, tentative_mask) if s]
    
    print(f"Confirmed Features: {len(selected_features)}")
    print(f"Tentative Features: {len(tentative_features)}")
    
    # Save selected features
    final_features = selected_features + tentative_features # Be generous
    if len(final_features) < 10:
        print("Warning: Boruta selected too few features. Using Top 50 by importance instead.")
        # Fallback to RF importance
        rf.fit(X_train, y_train)
        imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
        final_features = imp.head(50).index.tolist()
        
    with open('features_selected.json', 'w') as f:
        json.dump(final_features, f, indent=4)
    print("Saved features_selected.json")
    
    # --- STEP 2: OPTUNA TUNING ---
    print("\n--- Step 2: Optuna Tuning ---")
    
    # We need to rank features for the Siamese net (it likes Top N)
    # Let's get feature importance from XGBoost on the SELECTED features
    xgb_ranker = xgb.XGBClassifier(n_jobs=-1, random_state=42)
    xgb_ranker.fit(X_train[final_features], y_train)
    importances = pd.Series(xgb_ranker.feature_importances_, index=final_features).sort_values(ascending=False)
    ranked_features = importances.index.tolist()
    
    def objective(trial):
        # 1. XGBoost Params
        xgb_params = {
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
            'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
            'n_jobs': -1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        # Train XGB
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train[final_features], y_train)
        xgb_probs = model.predict_proba(X_test[final_features])[:, 1]
        
        # 2. Siamese Params
        # Optimize how many top features to use
        n_top_feats = trial.suggest_int('siamese_n_features', 30, 100)
        siamese_feats = ranked_features[:n_top_feats]
        
        siamese_lr = trial.suggest_float('siamese_lr', 1e-4, 1e-2, log=True)
        siamese_hidden = trial.suggest_categorical('siamese_hidden_dim', [32, 64, 128])
        
        # Prepare Data
        f1_tr, f2_tr, in_dim, _ = prepare_siamese_data(X_train, siamese_feats)
        f1_te, f2_te, _, _ = prepare_siamese_data(X_test, siamese_feats)
        
        # Scale
        scaler = StandardScaler()
        scaler.fit(np.concatenate([f1_tr, f2_tr]))
        f1_tr = scaler.transform(f1_tr)
        f2_tr = scaler.transform(f2_tr)
        f1_te = scaler.transform(f1_te)
        f2_te = scaler.transform(f2_te)
        
        # Train Siamese (Quickly - 5 epochs)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = SiameseMatchupNet(in_dim, hidden_dim=siamese_hidden).to(device)
        optimizer = optim.Adam(net.parameters(), lr=siamese_lr)
        
        tr_ds = TensorDataset(torch.FloatTensor(f1_tr), torch.FloatTensor(f2_tr), torch.FloatTensor(y_train))
        loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
        
        for _ in range(5): # Short training for tuning
            net.train()
            for b1, b2, by in loader:
                b1, b2, by = b1.to(device), b2.to(device), by.to(device)
                optimizer.zero_grad()
                loss = symmetric_loss(net, b1, b2, by)
                loss.backward()
                optimizer.step()
                
        net.eval()
        with torch.no_grad():
            t1 = torch.FloatTensor(f1_te).to(device)
            t2 = torch.FloatTensor(f2_te).to(device)
            siam_probs = net(t1, t2).cpu().numpy()
            
        # Ensemble
        w = trial.suggest_float('ensemble_weight', 0.0, 1.0)
        ens_probs = w * xgb_probs + (1 - w) * siam_probs
        
        acc = accuracy_score(y_test, (ens_probs > 0.5).astype(int))
        return acc
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30) # 30 trials for speed
    
    print("\nBest Params:")
    print(study.best_params)
    print(f"Best Accuracy: {study.best_value:.4f}")
    
    # Save params
    with open('params_optimized.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print("Saved params_optimized.json")

if __name__ == "__main__":
    run_optimization()
