import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# Import shared architecture
from models import SiameseMatchupNet, symmetric_loss, prepare_siamese_data

def run_time_split_verification():
    print("=== Starting Time-Split Verification (Holdout: 2024+) ===")
    
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists('data/training_data.csv'):
        print("Error: data/training_data.csv not found.")
        return

    df = pd.read_csv('data/training_data.csv')
    
    # Ensure date column
    if 'event_date' not in df.columns:
        print("Error: 'event_date' column missing.")
        return
        
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    with open('features.json', 'r') as f:
        features = json.load(f)
    print(f"Using {len(features)} features.")
    
    with open('params.json', 'r') as f:
        best = json.load(f)
        params = best['best_params']
    print("Loaded hyperparameters.")
    
    # 2. Split by Date
    split_date = '2025-01-01'
    train_df = df[df['event_date'] < split_date].copy()
    test_df = df[df['event_date'] >= split_date].copy()
    
    print(f"Training Set: {len(train_df)} fights (Pre-2025)")
    print(f"Holdout Set: {len(test_df)} fights (2025)")
    
    # Check for missing odds
    missing_odds = test_df[(test_df['f_1_odds'] == 0) | (test_df['f_1_odds'].isna()) | 
                           (test_df['f_2_odds'] == 0) | (test_df['f_2_odds'].isna())]
    print(f"Missing Odds in Holdout: {len(missing_odds)} / {len(test_df)} ({len(missing_odds)/len(test_df)*100:.1f}%)")
    
    if len(test_df) == 0:
        print("Error: No fights found in 2024+.")
        return
    
    # Prepare X and y
    X_train = train_df[[c for c in features if c in train_df.columns]].fillna(0)
    y_train = train_df['target'].values
    
    X_test = test_df[[c for c in features if c in test_df.columns]].fillna(0)
    y_test = test_df['target'].values
    
    # 3. Train XGBoost
    print("Training XGBoost...")
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
    
    # 4. Train Siamese
    print("Training Siamese Network...")
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
        
    # 5. Ensemble
    w = params['ensemble_xgb_weight']
    ens_probs = w * xgb_probs + (1 - w) * siamese_probs
    ens_preds = (ens_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, ens_preds)
    ll = log_loss(y_test, ens_probs)
    
    print(f"\n=== Holdout Results (2024-2025) ===")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Overall Log Loss: {ll:.4f}")
    print(f"Number of Fights: {len(y_test)}")
    
    # Evaluate on valid odds only
    valid_mask = (X_test['f_1_odds'] != 0) & (X_test['f_2_odds'] != 0)
    if valid_mask.sum() > 0:
        y_valid = y_test[valid_mask]
        preds_valid = ens_preds[valid_mask]
        acc_valid = accuracy_score(y_valid, preds_valid)
        print(f"\n=== Valid Odds Subset ({valid_mask.sum()} fights) ===")
        print(f"Accuracy: {acc_valid:.4f}")
    else:
        print("\nNo fights with valid odds found.")

if __name__ == "__main__":
    run_time_split_verification()
