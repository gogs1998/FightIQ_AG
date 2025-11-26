import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from models import SiameseMatchupNet, symmetric_loss, prepare_siamese_data

def run_debug_split():
    print("=== Debugging Accuracy Drop (2024 vs 2025) ===")
    
    # 1. Load Data
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    with open('features.json', 'r') as f: features = json.load(f)
    with open('params.json', 'r') as f: params = json.load(f)['best_params']
    
    # 2. Define Splits
    # Train: < 2024
    # Test 2024: >= 2024 & < 2025
    # Test 2025: >= 2025
    
    train_df = df[df['event_date'] < '2024-01-01'].copy()
    test_2024 = df[(df['event_date'] >= '2024-01-01') & (df['event_date'] < '2025-01-01')].copy()
    test_2025 = df[df['event_date'] >= '2025-01-01'].copy()
    
    print(f"Train Set (Pre-2024): {len(train_df)}")
    print(f"Test Set (2024): {len(test_2024)}")
    print(f"Test Set (2025): {len(test_2025)}")
    
    # 3. Train Model (on Pre-2024)
    print("\nTraining model on Pre-2024 data...")
    X_train = train_df[[c for c in features if c in train_df.columns]].fillna(0)
    y_train = train_df['target'].values
    
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
    
    # Siamese
    f1_train, f2_train, input_dim, _ = prepare_siamese_data(X_train, features)
    scaler = StandardScaler()
    combined_train = np.concatenate([f1_train, f2_train], axis=0)
    scaler.fit(combined_train)
    f1_train = scaler.transform(f1_train)
    f2_train = scaler.transform(f2_train)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
    train_ds = TensorDataset(torch.FloatTensor(f1_train), torch.FloatTensor(f2_train), torch.FloatTensor(y_train))
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
    
    # 4. Evaluate 2024
    def evaluate(test_df, name):
        X_test = test_df[[c for c in features if c in test_df.columns]].fillna(0)
        y_test = test_df['target'].values
        
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        
        f1_test, f2_test, _, _ = prepare_siamese_data(X_test, features)
        f1_test = scaler.transform(f1_test)
        f2_test = scaler.transform(f2_test)
        
        with torch.no_grad():
            t_f1 = torch.FloatTensor(f1_test).to(device)
            t_f2 = torch.FloatTensor(f2_test).to(device)
            siamese_probs = siamese_model(t_f1, t_f2).cpu().numpy()
            
        w = params['ensemble_xgb_weight']
        ens_probs = w * xgb_probs + (1 - w) * siamese_probs
        ens_preds = (ens_probs > 0.5).astype(int)
        
        acc = accuracy_score(y_test, ens_preds)
        print(f"\n=== {name} Results ===")
        print(f"Accuracy: {acc:.4f} ({int(acc*len(y_test))}/{len(y_test)})")
        
        # Valid Odds Only
        valid_mask = (test_df['f_1_odds'] > 1.0) & (test_df['f_2_odds'] > 1.0)
        if valid_mask.sum() > 0:
            acc_valid = accuracy_score(y_test[valid_mask], ens_preds[valid_mask])
            print(f"Accuracy (Valid Odds): {acc_valid:.4f} ({valid_mask.sum()} fights)")
        else:
            print("No valid odds.")

    evaluate(test_2024, "2024")
    evaluate(test_2025, "2025")

if __name__ == "__main__":
    run_debug_split()
