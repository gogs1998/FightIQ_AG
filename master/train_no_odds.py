import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

# Import shared architecture
from models import SiameseMatchupNet, symmetric_loss, prepare_siamese_data

def train_no_odds_pipeline():
    print("=== Starting No-Odds (Fantasy) Training Pipeline ===")
    
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists('data/training_data.csv'):
        print("Error: data/training_data.csv not found.")
        return

    df = pd.read_csv('data/training_data.csv')
    
    # 2. Load Features & Remove Odds
    with open('features.json', 'r') as f:
        all_features = json.load(f)
    
    # Filter out odds features
    odds_keywords = ['odds', 'implied_prob']
    features = [f for f in all_features if not any(k in f.lower() for k in odds_keywords)]
    
    print(f"Original features: {len(all_features)}")
    print(f"No-Odds features: {len(features)}")
    
    # 3. Load Hyperparameters (Reuse optimized params)
    with open('params.json', 'r') as f:
        best = json.load(f)
        params = best['best_params']
    
    # 4. Prepare Data
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X_df = df[[c for c in features if c in df.columns]]
    X_df = X_df.fillna(0)
    y = df['target'].values
    
    # Time-based Split (Cutoff: 2024-01-01)
    split_date = '2024-01-01'
    mask_train = df['event_date'] < split_date
    mask_test = df['event_date'] >= split_date
    
    X_train = X_df[mask_train]
    X_test = X_df[mask_test]
    y_train = y[mask_train]
    y_test = y[mask_test]
    
    print(f"Train Set: {len(X_train)}")
    print(f"Test Set:  {len(X_test)}")
    
    # 5. Train XGBoost
    print("Training XGBoost (No Odds)...")
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
    xgb_acc = accuracy_score(y_test, (xgb_probs > 0.5).astype(int))
    print(f"XGBoost (No Odds) Accuracy: {xgb_acc:.4f}")
    
    # 6. Train Siamese
    print("Training Siamese Network (No Odds)...")
    f1_train, f2_train, input_dim, siamese_cols = prepare_siamese_data(X_train, features)
    f1_test, f2_test, _, _ = prepare_siamese_data(X_test, features)
    
    scaler = StandardScaler()
    combined_train = np.concatenate([f1_train, f2_train], axis=0)
    scaler.fit(combined_train)
    
    f1_train = scaler.transform(f1_train)
    f2_train = scaler.transform(f2_train)
    f1_test = scaler.transform(f1_test)
    f2_test = scaler.transform(f2_test)
    
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
    with torch.no_grad():
        t_f1 = torch.FloatTensor(f1_test).to(device)
        t_f2 = torch.FloatTensor(f2_test).to(device)
        siamese_probs = siamese_model(t_f1, t_f2).cpu().numpy()
        
    siamese_acc = accuracy_score(y_test, (siamese_probs > 0.5).astype(int))
    print(f"Siamese (No Odds) Accuracy: {siamese_acc:.4f}")
    
    # 7. Ensemble
    w = params['ensemble_xgb_weight']
    ens_probs = w * xgb_probs + (1 - w) * siamese_probs
    ens_preds = (ens_probs > 0.5).astype(int)
    final_acc = accuracy_score(y_test, ens_preds)
    
    print(f"\n=== Final No-Odds Model Performance ===")
    print(f"Accuracy: {final_acc:.4f}")
    
    # 8. Save Artifacts
    print("Saving artifacts to models/no_odds/...")
    os.makedirs('models/no_odds', exist_ok=True)
    
    joblib.dump(xgb_model, 'models/no_odds/xgb_no_odds.pkl')
    torch.save(siamese_model.state_dict(), 'models/no_odds/siamese_no_odds.pth')
    joblib.dump(scaler, 'models/no_odds/siamese_scaler.pkl')
    
    with open('models/no_odds/siamese_cols.json', 'w') as f:
        json.dump(siamese_cols, f)
        
    print("Done.")

if __name__ == "__main__":
    train_no_odds_pipeline()
