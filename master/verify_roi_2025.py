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

def evaluate_strategies(df, preds, probs):
    """
    Evaluate multiple betting strategies including Kelly Criterion.
    """
    strategies = {
        "Flat (All Bets)": lambda row, p, prob, bk: (True, 10.0),
        "Value (Edge > 5%)": lambda row, p, prob, bk: is_value_bet(row, p, prob, 0.05),
        "Kelly (Full)": lambda row, p, prob, bk: kelly_bet(row, p, prob, bk, 1.0),
        "Kelly (1/4)": lambda row, p, prob, bk: kelly_bet(row, p, prob, bk, 0.25),
        "Kelly (1/8)": lambda row, p, prob, bk: kelly_bet(row, p, prob, bk, 0.125),
    }
    
    results = {}
    
    for name, strategy_fn in strategies.items():
        bankroll = 1000.0
        wagered = 0.0
        returned = 0.0
        wins = 0
        losses = 0
        bets_placed = 0
        
        if name == "Kelly (Full)":
            bet_log = []
        
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            row = df.iloc[i]
            
            # Determine target and odds
            if pred == 1:
                odds = row['f_1_odds']
                actual_winner = row['target'] # 1
                won_bet = (actual_winner == 1)
                my_prob = prob
                target_name = row['f_1_name']
                opponent_name = row['f_2_name']
            else:
                odds = row['f_2_odds']
                actual_winner = row['target'] # 0
                won_bet = (actual_winner == 0)
                my_prob = 1 - prob
                target_name = row['f_2_name']
                opponent_name = row['f_1_name']
                
            if odds <= 1.0: continue
            
            # Check strategy
            should_bet, size = strategy_fn(row, pred, my_prob, bankroll)
            
            # Cap bet size to bankroll (cannot bet more than we have)
            if size > bankroll:
                size = bankroll
            
            if should_bet and size > 0:
                bets_placed += 1
                wagered += size
                if won_bet:
                    payout = size * odds
                    returned += payout
                    profit = payout - size
                    bankroll += profit
                    wins += 1
                    res_str = "WIN"
                else:
                    profit = -size
                    bankroll -= size
                    losses += 1
                    res_str = "LOSS"
                
                if name == "Kelly (Full)":
                    bet_log.append({
                        "Date": str(row['event_date']).split(' ')[0],
                        "Fighter": target_name,
                        "Opponent": opponent_name,
                        "Prob": f"{my_prob:.1%}",
                        "Odds": odds,
                        "Stake": f"${size:.2f}",
                        "Result": res_str,
                        "P/L": f"${profit:.2f}",
                        "Bankroll": f"${bankroll:.2f}"
                    })
        
        if name == "Kelly (Full)":
            pd.DataFrame(bet_log).to_csv('kelly_full_bets.csv', index=False)
            print(f"Saved {len(bet_log)} bets to kelly_full_bets.csv")
        
        net_profit = returned - wagered
        roi = (net_profit / wagered * 100) if wagered > 0 else 0.0
        
        results[name] = {
            "bets": bets_placed,
            "wagered": wagered,
            "profit": net_profit,
            "roi": roi,
            "win_rate": (wins/bets_placed*100) if bets_placed > 0 else 0,
            "final_bankroll": bankroll
        }
        
    return results

def is_value_bet(row, pred, prob, margin):
    if pred == 1:
        odds = row['f_1_odds']
    else:
        odds = row['f_2_odds']
        
    if odds <= 1.0: return False, 0.0
    
    implied = 1.0 / odds
    edge = prob - implied
    
    if edge > margin:
        return True, 10.0
    return False, 0.0

def kelly_bet(row, pred, prob, bankroll, fraction=1.0):
    if pred == 1:
        odds = row['f_1_odds']
    else:
        odds = row['f_2_odds']
        
    if odds <= 1.0: return False, 0.0
    
    b = odds - 1.0
    q = 1.0 - prob
    
    # Kelly Formula: f* = (bp - q) / b
    f_star = (b * prob - q) / b
    
    if f_star > 0:
        stake = bankroll * f_star * fraction
        # Safety cap: never bet more than 20% of bankroll on one fight
        max_stake = bankroll * 0.20
        if stake > max_stake: stake = max_stake
        return True, stake
        
    return False, 0.0

def run_roi_analysis():
    print("=== Starting ROI Analysis (Train: Pre-2025, Test: 2025 Valid Odds) ===")
    
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
    
    # 2. Split by Date (2025 Holdout)
    split_date = '2025-01-01'
    train_df = df[df['event_date'] < split_date].copy()
    test_df_raw = df[df['event_date'] >= split_date].copy()
    
    print(f"Training Set: {len(train_df)} fights (Pre-2025)")
    print(f"Raw Holdout Set: {len(test_df_raw)} fights (2025)")
    
    # 3. Filter Test Set for Valid Odds
    test_df = test_df_raw[(test_df_raw['f_1_odds'] > 1.0) & (test_df_raw['f_2_odds'] > 1.0)].copy()
    print(f"Filtered Holdout Set (Valid Odds): {len(test_df)} fights")
    
    if len(test_df) == 0:
        print("Error: No fights with valid odds in 2025.")
        return
    
    # Prepare X and y
    X_train = train_df[[c for c in features if c in train_df.columns]].fillna(0)
    y_train = train_df['target'].values
    
    X_test = test_df[[c for c in features if c in test_df.columns]].fillna(0)
    y_test = test_df['target'].values
    
    # 4. Train XGBoost
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
    
    # 5. Train Siamese
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
        
    # 6. Ensemble
    w = params['ensemble_xgb_weight']
    ens_probs = w * xgb_probs + (1 - w) * siamese_probs
    ens_preds = (ens_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, ens_preds)
    
    # 7. ROI Calculation
    results = evaluate_strategies(test_df, ens_preds, ens_probs)
    
    print(f"\n=== 2025 ROI Analysis (Valid Odds Only) ===")
    print(f"{'Strategy':<20} | {'Bets':<5} | {'Win Rate':<8} | {'Profit':<10} | {'ROI':<8} | {'Final Bankroll':<15}")
    print("-" * 90)
    
    for name, res in results.items():
        print(f"{name:<20} | {res['bets']:<5} | {res['win_rate']:>6.1f}% | ${res['profit']:>8.2f} | {res['roi']:>6.2f}% | ${res['final_bankroll']:>10.2f}")

if __name__ == "__main__":
    run_roi_analysis()
