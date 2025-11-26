import pandas as pd
import numpy as np
import json
import os
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Siamese Network Architecture
class SiameseMatchupNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
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

def load_data_with_features():
    """Load data and selected features"""
    print("Loading data...")
    df = pd.read_csv('v2/data/training_data_v2.csv')
    
    # Load selected features (SAFE LIST)
    with open('features_elo.json', 'r') as f:
        selected_features = json.load(f)
    
    # Prepare X and y
    exclude_cols = ['target', 'winner', 'winner_encoded', 'f_1_name', 'f_2_name', 'event_date', 'fight_id', 'event_id', 'weight_class', 'method', 'round', 'time']
    
    # Filter to selected features only
    X_df = df[[c for c in selected_features if c in df.columns]]
    X_df = X_df.fillna(0)
    
    y = df['target'].values
    
    return X_df, y, selected_features

def prepare_siamese_data(X_df, features):
    """
    Prepare data for Siamese network using robust pair finding from test_ensemble_all.py
    """
    # Identify pairs from features
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
    # print(f"Found {len(pairs)} feature pairs.")
    
    numeric_pairs = []
    for c1, c2 in pairs:
        # Check if columns exist in X_df (they should, but safe check)
        if c1 in X_df.columns and c2 in X_df.columns:
            numeric_pairs.append((c1, c2))
            
    f1_feats = [p[0] for p in numeric_pairs]
    f2_feats = [p[1] for p in numeric_pairs]
    
    if not f1_feats:
        # Fallback if no pairs found (e.g. only diff features)
        return np.zeros((len(X_df), 1)), np.zeros((len(X_df), 1)), 1

    f1_data = X_df[f1_feats].values
    f2_data = X_df[f2_feats].values
    
    return f1_data, f2_data, len(f1_feats)

def objective(trial):
    """Optuna objective function"""
    X, y, features = load_data_with_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Hyperparameters to optimize
    xgb_max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
    xgb_learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
    xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
    xgb_min_child_weight = trial.suggest_int('xgb_min_child_weight', 1, 10)
    xgb_subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0)
    xgb_colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0)
    
    siamese_hidden_dim = trial.suggest_int('siamese_hidden_dim', 64, 256, step=64)
    siamese_epochs = trial.suggest_int('siamese_epochs', 10, 30, step=5)
    siamese_lr = trial.suggest_float('siamese_lr', 0.0001, 0.01, log=True)
    
    ensemble_xgb_weight = trial.suggest_float('ensemble_xgb_weight', 0.3, 0.7)
    
    # Train XGBoost
    print(f"Trial {trial.number}: Training XGBoost...")
    xgb_model = XGBClassifier(
        max_depth=xgb_max_depth,
        learning_rate=xgb_learning_rate,
        n_estimators=xgb_n_estimators,
        min_child_weight=xgb_min_child_weight,
        subsample=xgb_subsample,
        colsample_bytree=xgb_colsample_bytree,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Train Siamese Network
    print(f"Trial {trial.number}: Training Siamese Network...")
    f1_train, f2_train, input_dim = prepare_siamese_data(X_train, features)
    f1_test, f2_test, _ = prepare_siamese_data(X_test, features)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=siamese_hidden_dim).to(device)
    
    # Prepare DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(f1_train),
        torch.FloatTensor(f2_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    optimizer = torch.optim.Adam(siamese_model.parameters(), lr=siamese_lr)
    criterion = nn.BCELoss()
    
    for epoch in range(siamese_epochs):
        siamese_model.train()
        for batch_f1, batch_f2, batch_y in train_loader:
            batch_f1, batch_f2, batch_y = batch_f1.to(device), batch_f2.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = siamese_model(batch_f1, batch_f2)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Get Siamese predictions
    siamese_model.eval()
    with torch.no_grad():
        f1_test_tensor = torch.FloatTensor(f1_test).to(device)
        f2_test_tensor = torch.FloatTensor(f2_test).to(device)
        siamese_probs = siamese_model(f1_test_tensor, f2_test_tensor).cpu().numpy()
    
    # Ensemble
    ensemble_probs = ensemble_xgb_weight * xgb_probs + (1 - ensemble_xgb_weight) * siamese_probs
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, ensemble_preds)
    logloss = log_loss(y_test, ensemble_probs)
    
    print(f"Trial {trial.number}: Accuracy={accuracy:.4f}, LogLoss={logloss:.4f}")
    
    # Report intermediate value for pruning
    trial.report(accuracy, siamese_epochs)
    
    # Handle pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    return accuracy

def run_optuna_study(n_trials=100):
    """Run Optuna optimization"""
    print(f"Starting Optuna study with {n_trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='fightiq_optimization',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n=== Optimization Complete ===")
    print(f"Best Accuracy: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    os.makedirs('experimental', exist_ok=True)
    with open('experimental/optuna_best_params.json', 'w') as f:
        json.dump({
            'best_accuracy': study.best_value,
            'best_params': study.best_params
        }, f, indent=4)
    
    print("\nSaved best parameters to experimental/optuna_best_params.json")
    
    return study

if __name__ == "__main__":
    study = run_optuna_study(n_trials=100)
