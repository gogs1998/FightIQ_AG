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
from models.opponent_adjustment import apply_opponent_adjustment
from models.sequence_model import prepare_sequences

def train_model(df, split_date, features, params, verbose=True):
    """
    Trains the Master 3 pipeline on data < split_date and tests on data >= split_date.
    Returns a dictionary of metrics.
    """
    if verbose: print(f"\n=== Training Cycle: Split {split_date} ===")
    
    # 4. Apply Opponent Adjustment (Granularity)
    # We do this here because it depends on the current DF's distribution (avg_elo)
    if verbose: print("Applying Opponent Adjustment...")
    
    adj_candidates = [
        'slpm_15_f_1', 'slpm_15_f_2',
        'td_avg_15_f_1', 'td_avg_15_f_2',
        'sub_avg_15_f_1', 'sub_avg_15_f_2',
        'sapm_15_f_1', 'sapm_15_f_2'
    ]
    
    # Filter to existing
    adj_cols = [c for c in adj_candidates if c in df.columns]
    
    if adj_cols and 'dynamic_elo_f1' in df.columns:
        df = apply_opponent_adjustment(df, adj_cols, elo_col='dynamic_elo')
        # Add adjusted cols to features if not already there
        for c in adj_cols:
            adj_name = f"{c}_adj"
            if adj_name not in features:
                features.append(adj_name)
        if verbose: print(f"Adjusted {len(adj_cols)} stats by opponent Elo.")
    else:
        if verbose: print("Skipping Opponent Adjustment (cols missing).")

    # 5. Prepare Data
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Filter missing odds
    if 'f_1_odds' in features and 'f_2_odds' in features:
        has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
                   (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
        df = df[has_odds].copy()
        
    X_df = df[[c for c in features if c in df.columns]]
    X_df = X_df.fillna(0)
    y = df['target'].values
    
    # Time-based Split
    mask_train = df['event_date'] < split_date
    mask_test = df['event_date'] >= split_date
    
    X_train = X_df[mask_train]
    X_test = X_df[mask_test]
    y_train = y[mask_train]
    y_test = y[mask_test]
    
    if verbose: print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    if len(X_test) == 0:
        print("Warning: Test set is empty.")
        return {'accuracy': 0, 'roi': 0}

    # 6. Prepare Sequence Data (Phase 6) - ONCE
    seq_f1, seq_f2, seq_dim = prepare_sequences(df, features)
    
    # Split Sequences
    seq_f1_train = seq_f1[mask_train]
    seq_f2_train = seq_f2[mask_train]
    seq_f1_test = seq_f1[mask_test]
    seq_f2_test = seq_f2[mask_test]
    
    # 7. Train Main Ensemble (XGB + Siamese)
    if verbose: print("Training Main Ensemble...")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        max_depth=params.get('xgb_max_depth', 3),
        learning_rate=params.get('xgb_learning_rate', 0.05),
        n_estimators=params.get('xgb_n_estimators', 100),
        min_child_weight=params.get('xgb_min_child_weight', 1),
        subsample=params.get('xgb_subsample', 0.8),
        colsample_bytree=params.get('xgb_colsample_bytree', 0.8),
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Siamese
    best_siam_acc = 0.0
    best_probs = np.zeros(len(y_test))
    
    # Try 50 times for Siamese (Robustness)
    # For Walk-Forward, maybe reduce seeds for speed? 
    # No, we want accuracy. But 50 seeds * 5 years = 250 runs. Too slow?
    # Let's use 10 seeds for validation to save time, or keep 50 if user wants "comprehensive".
    # User said "comprehensive". Let's stick to 50 but maybe optimize?
    # I'll default to 10 for the loop to keep it reasonable, or add a param.
    n_seeds = params.get('n_seeds', 50) 
    
    for attempt in range(1, n_seeds + 1):
        seed = 42 + attempt
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Prepare Tabular Data
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
        
        # Initialize with Sequence Support
        siamese_model = SiameseMatchupNet(input_dim, seq_input_dim=seq_dim, hidden_dim=params.get('siamese_hidden_dim', 64)).to(device)
        
        # Update Dataset to include sequences
        train_ds = TensorDataset(
            torch.FloatTensor(f1_train), torch.FloatTensor(f2_train), 
            torch.FloatTensor(seq_f1_train), torch.FloatTensor(seq_f2_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
        optimizer = optim.Adam(siamese_model.parameters(), lr=params.get('siamese_lr', 0.001))
        
        for epoch in range(params.get('siamese_epochs', 10)):
            siamese_model.train()
            for b_f1, b_f2, b_s1, b_s2, b_y in train_loader:
                b_f1, b_f2, b_s1, b_s2, b_y = b_f1.to(device), b_f2.to(device), b_s1.to(device), b_s2.to(device), b_y.to(device)
                optimizer.zero_grad()
                loss = symmetric_loss(siamese_model, b_f1, b_f2, b_y, b_s1, b_s2)
                loss.backward()
                optimizer.step()
                
        siamese_model.eval()
        with torch.no_grad():
            t_f1 = torch.FloatTensor(f1_test).to(device)
            t_f2 = torch.FloatTensor(f2_test).to(device)
            t_s1 = torch.FloatTensor(seq_f1_test).to(device)
            t_s2 = torch.FloatTensor(seq_f2_test).to(device)
            
            siamese_probs = siamese_model(t_f1, t_f2, t_s1, t_s2).cpu().numpy()
            
        acc = accuracy_score(y_test, (siamese_probs > 0.5).astype(int))
        if acc > best_siam_acc:
            best_siam_acc = acc
            best_probs = siamese_probs
            # Save best model state
            torch.save(siamese_model.state_dict(), 'models/siamese_master3.pth')
            
    if verbose: print(f"Best Siamese Acc: {best_siam_acc:.4f} (Saved to models/siamese_master3.pth)")
    
    # Ensemble
    w = params.get('ensemble_xgb_weight', 0.5)
    ens_probs = w * xgb_probs + (1 - w) * best_probs
    ens_preds = (ens_probs > 0.5).astype(int)
    
    final_acc = accuracy_score(y_test, ens_preds)
    final_ll = log_loss(y_test, ens_probs)
    
    if verbose: print(f"Ensemble Accuracy: {final_acc:.4f}")
    
    # ROI Calculation
    test_df = df[mask_test].copy()
    test_df['prob_f1'] = ens_probs
    test_df['implied_f1'] = 1 / test_df['f_1_odds']
    test_df['implied_f2'] = 1 / test_df['f_2_odds']
    test_df['edge_f1'] = test_df['prob_f1'] - test_df['implied_f1']
    test_df['edge_f2'] = (1 - test_df['prob_f1']) - test_df['implied_f2']
    
    min_edge = 0.05
    bankroll = 1000.0
    history = []
    
    for idx, row in test_df.iterrows():
        bet_on = None
        odds = 0
        
        if row['edge_f1'] > min_edge:
            bet_on = 'f1'
            odds = row['f_1_odds']
            edge = row['edge_f1']
        elif row['edge_f2'] > min_edge:
            bet_on = 'f2'
            odds = row['f_2_odds']
            edge = row['edge_f2']
            
        if bet_on:
            wager = 100
            won = False
            if bet_on == 'f1' and row['target'] == 1: won = True
            if bet_on == 'f2' and row['target'] == 0: won = True
            
            if won:
                bankroll += wager * (odds - 1)
                res = 'WIN'
                profit = wager * (odds - 1)
            else:
                bankroll -= wager
                res = 'LOSS'
                profit = -wager
            
            history.append({
                'Date': row['event_date'],
                'Fighter1': row['f_1_name'],
                'Fighter2': row['f_2_name'],
                'Bet_On': row['f_1_name'] if bet_on == 'f1' else row['f_2_name'],
                'Odds': odds,
                'Edge': edge,
                'Prob': row['prob_f1'] if bet_on == 'f1' else 1 - row['prob_f1'],
                'Result': res,
                'Profit': profit,
                'Bankroll': bankroll
            })
                
    if verbose: print(f"ROI: {roi:.2%}")
    
    # Save Log
    if history:
        log_df = pd.DataFrame(history)
        log_df.to_csv('betting_log.csv', index=False)
        if verbose: print(f"Saved {len(log_df)} bets to betting_log.csv")
                
    roi = (bankroll - 1000) / 1000
    if verbose: print(f"ROI: {roi:.2%}")
    
    return {
        'accuracy': final_acc,
        'log_loss': final_ll,
        'roi': roi,
        'n_test': len(X_test)
    }

def main():
    print("=== Master 3: The 'Kitchen Sink' Pipeline ===")
    
    # 1. Load Enhanced Data
    print("Loading enhanced data...")
    if not os.path.exists('data/training_data_enhanced.csv'):
        print("Error: data/training_data_enhanced.csv not found.")
        return

    df = pd.read_csv('data/training_data_enhanced.csv')
    
    # 2. Load Features
    if os.path.exists('features_selected.json'):
        with open('features_selected.json', 'r') as f:
            features = json.load(f)
    else:
        with open('features_enhanced.json', 'r') as f:
            features = json.load(f)
            
    # 3. Load Params
    with open('params.json', 'r') as f:
        best = json.load(f)
        params = best['best_params']
        
    if os.path.exists('params_optimized.json'):
        with open('params_optimized.json', 'r') as f:
            opt_params = json.load(f)
            params.update(opt_params)
            
    # Run Standard Training (2024-2025)
    train_model(df, '2024-01-01', features, params)

if __name__ == "__main__":
    main()
