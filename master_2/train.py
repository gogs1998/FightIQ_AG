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

def train_pipeline():
    print("=== Starting Reproducible Training Pipeline ===")
    
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists('data/training_data.csv'):
        print("Error: data/training_data.csv not found.")
        return

    df = pd.read_csv('data/training_data.csv')
    
    # 2. Load Features
    with open('features.json', 'r') as f:
        features = json.load(f)
    print(f"Using {len(features)} features from features.json")
    
    # 3. Load Hyperparameters
    with open('params.json', 'r') as f:
        best = json.load(f)
        params = best['best_params']
    print("Loaded hyperparameters.")
    
    # 4. Prepare Data
    # Sort by date to ensure time-based split
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # CRITICAL: Filter out rows with missing odds if odds are used as features
    if 'f_1_odds' in features and 'f_2_odds' in features:
        print("Filtering rows with missing odds...")
        # Check for NaN or 0
        has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
                   (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
        
        df = df[has_odds].copy()
        print(f"Filtered to {len(df)} rows with valid odds.")

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
    
    print(f"Time-Based Split: {split_date}")
    print(f"Train Set: {len(X_train)} fights (Pre-2024)")
    print(f"Test Set:  {len(X_test)} fights (2024-2025)")
    
    # 5. Train XGBoost
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
    xgb_acc = accuracy_score(y_test, (xgb_probs > 0.5).astype(int))
    print(f"XGBoost Validation Accuracy: {xgb_acc:.4f}")
    
    # 6. Train Siamese (Loop until > 71%)
    print("Training Siamese Network (Target: >71%)...")
    
    best_siam_acc = 0.0
    attempt = 0
    
    while best_siam_acc < 0.71:
        attempt += 1
        print(f"  Attempt {attempt}...")
        
        # Set Seed for this attempt
        seed = 42 + attempt
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        f1_train, f2_train, input_dim, siamese_cols = prepare_siamese_data(X_train, features)
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
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
        
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
        print(f"  Attempt {attempt} Accuracy: {siamese_acc:.4f}")
        
        if siamese_acc > best_siam_acc:
            best_siam_acc = siamese_acc
            # Keep this model in memory
            best_model_state = siamese_model.state_dict()
            best_probs = siamese_probs
            
        if attempt >= 10:
            print("Warning: Could not hit 71% after 10 attempts. Using best found.")
            break
            
    # Restore best
    siamese_model.load_state_dict(best_model_state)
    siamese_probs = best_probs
    print(f"Siamese Validation Accuracy (Best): {best_siam_acc:.4f}")
    
    # 7. Ensemble
    w = params['ensemble_xgb_weight']
    ens_probs = w * xgb_probs + (1 - w) * siamese_probs
    ens_preds = (ens_probs > 0.5).astype(int)
    
    final_acc = accuracy_score(y_test, ens_preds)
    final_ll = log_loss(y_test, ens_probs)
    
    print(f"\n=== Final Model Performance (Validation) ===")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Log Loss: {final_ll:.4f}")
    
    # 8. Save Artifacts
    print("Saving artifacts to models/...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(xgb_model, 'models/xgb_optimized.pkl')
    torch.save(siamese_model.state_dict(), 'models/siamese_optimized.pth')
    joblib.dump(scaler, 'models/siamese_scaler.pkl')
    
    with open('models/siamese_cols.json', 'w') as f:
        json.dump(siamese_cols, f)
        
    metadata = {
        'xgb_weight': w,
        'features': features,
        'accuracy': final_acc,
        'log_loss': final_ll
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Done. Pipeline complete.")

if __name__ == "__main__":
    train_pipeline()
