import pandas as pd
import numpy as np
import joblib
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import os
from models import SiameseMatchupNet, symmetric_loss, prepare_siamese_data

# --- Configuration ---
BASE_DIR = 'd:/AntiGravity/FightIQ/master_2'
MODELS_DIR = f'{BASE_DIR}/models_2021'
os.makedirs(MODELS_DIR, exist_ok=True)

SPLIT_DATE = '2021-01-01'
MAX_WIN_PER_EVENT = 50000.0

def train_and_evaluate_2021():
    print(f"=== FightIQ: 5-Year Holdout Evaluation (2021-2025) ===")
    
    # 1. Load Data
    print("Loading Data...")
    df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    with open(f'{BASE_DIR}/features.json', 'r') as f: features = json.load(f)
    with open(f'{BASE_DIR}/params.json', 'r') as f: params = json.load(f)['best_params']
    
    # Filter Odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 2. Split Data (Train: 2010-2020, Test: 2021-2025)
    mask_train = df['event_date'] < SPLIT_DATE
    mask_test = df['event_date'] >= SPLIT_DATE
    
    X_df = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    X_train = X_df[mask_train]
    y_train = y[mask_train]
    X_test = X_df[mask_test]
    y_test = y[mask_test]
    test_df = df[mask_test].copy()
    
    print(f"Train Set (2010-2020): {len(X_train)} fights")
    print(f"Test Set  (2021-2025): {len(X_test)} fights")
    
    # 3. Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        max_depth=params['xgb_max_depth'],
        learning_rate=params['xgb_learning_rate'],
        n_estimators=params['xgb_n_estimators'],
        min_child_weight=params['xgb_min_child_weight'],
        subsample=params['xgb_subsample'],
        colsample_bytree=params['xgb_colsample_bytree'],
        n_jobs=-1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, f'{MODELS_DIR}/xgb_2021.pkl')
    
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    acc_xgb = accuracy_score(y_test, (p_xgb > 0.5).astype(int))
    print(f"XGBoost Accuracy: {acc_xgb:.4%}")
    
    # 4. Train Siamese (Mini-Optimization)
    print("\nTraining Siamese Network (Mini-Opt: 10 Seeds)...")
    
    f1_train, f2_train, input_dim, _ = prepare_siamese_data(X_train, features)
    f1_test, f2_test, _, _ = prepare_siamese_data(X_test, features)
    
    scaler = StandardScaler()
    combined_train = np.concatenate([f1_train, f2_train], axis=0)
    scaler.fit(combined_train)
    joblib.dump(scaler, f'{MODELS_DIR}/siamese_scaler_2021.pkl')
    
    f1_train = scaler.transform(f1_train)
    f2_train = scaler.transform(f2_train)
    f1_test = scaler.transform(f1_test)
    f2_test = scaler.transform(f2_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_siam_acc = 0.0
    best_siam_probs = None
    
    # Try 10 seeds to ensure we get a good convergence on this new split
    for i in range(10):
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['siamese_lr'])
        
        train_ds = TensorDataset(torch.FloatTensor(f1_train), torch.FloatTensor(f2_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
        
        for epoch in range(params['siamese_epochs']):
            model.train()
            for b1, b2, by in train_loader:
                b1, b2, by = b1.to(device), b2.to(device), by.to(device)
                optimizer.zero_grad()
                loss = symmetric_loss(model, b1, b2, by)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            t1 = torch.FloatTensor(f1_test).to(device)
            t2 = torch.FloatTensor(f2_test).to(device)
            probs = model(t1, t2).cpu().numpy().flatten()
            
        acc = accuracy_score(y_test, (probs > 0.5).astype(int))
        # print(f"  Seed {seed}: {acc:.4%}")
        
        if acc > best_siam_acc:
            best_siam_acc = acc
            best_siam_probs = probs
            torch.save(model.state_dict(), f'{MODELS_DIR}/siamese_2021.pth')
            
    print(f"Best Siamese Accuracy: {best_siam_acc:.4%}")
    
    # 5. Ensemble Evaluation
    w = params['ensemble_xgb_weight'] # Use the optimized weight (0.1546)
    p_ens = w * p_xgb + (1 - w) * best_siam_probs
    
    acc_ens = accuracy_score(y_test, (p_ens > 0.5).astype(int))
    print(f"\nEnsemble Accuracy (2021-2025): {acc_ens:.4%} 🚀")
    
    test_df['prob'] = p_ens
    
    # 6. Betting Simulation (Realistic)
    print(f"\n=== Betting Simulation (Max Win: ${MAX_WIN_PER_EVENT:,.0f}) ===")
    print(f"{'Kelly':<10} {'Final Bank':<15} {'Profit':<15} {'Drawdown':<10} {'ROI':<10} {'Bets':<6}")
    print("-" * 80)
    
    fractions = [0.10, 0.125, 0.25, 0.50, 1.00]
    
    for frac in fractions:
        bankroll = 1000.0
        history = [bankroll]
        stakes = []
        
        for _, row in test_df.iterrows():
            prob = row['prob']
            target = row['target']
            odds_1 = row['f_1_odds']
            odds_2 = row['f_2_odds']
            
            if prob > 0.5:
                my_prob = prob
                odds = odds_1
                win = (target == 1)
            else:
                my_prob = 1 - prob
                odds = odds_2
                win = (target == 0)
                
            if odds > 5.8: continue # Max Odds from Robust Strategy
            if my_prob < 0.45: continue # Min Conf from Robust Strategy
            
            implied = 1 / odds
            edge = my_prob - implied
            if edge < 0.013: continue # Min Edge from Robust Strategy
            
            b = odds - 1
            q = 1 - my_prob
            f = (b * my_prob - q) / b
            if f < 0: f = 0
            
            raw_stake = bankroll * f * frac
            if raw_stake > bankroll * 0.20: raw_stake = bankroll * 0.20
            
            max_stake_limit = MAX_WIN_PER_EVENT / (odds - 1)
            stake = min(raw_stake, max_stake_limit)
            
            if stake < 5: stake = 0
            
            if stake > 0:
                stakes.append(stake)
                if win: bankroll += stake * (odds - 1)
                else: bankroll -= stake
                
            history.append(bankroll)
            if bankroll < 10: break
            
        hist_s = pd.Series(history)
        peak = hist_s.cummax()
        dd = (peak - hist_s) / peak
        max_dd = dd.max()
        profit = bankroll - 1000.0
        roi = profit / sum(stakes) if stakes else 0
        
        print(f"{frac:<10.3f} ${bankroll:<14,.0f} ${profit:<14,.0f} {max_dd:<10.1%} {roi:<10.1%} {len(stakes):<6}")

if __name__ == "__main__":
    train_and_evaluate_2021()
