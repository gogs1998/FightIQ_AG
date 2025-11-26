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

# --- Strategy Functions ---
def evaluate_strategies(df, preds, probs):
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
        bets_placed = 0
        
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            row = df.iloc[i]
            if pred == 1:
                odds = row['f_1_odds']
                actual_winner = row['target']
                won_bet = (actual_winner == 1)
                my_prob = prob
            else:
                odds = row['f_2_odds']
                actual_winner = row['target']
                won_bet = (actual_winner == 0)
                my_prob = 1 - prob
                
            if odds <= 1.0: continue
            
            should_bet, size = strategy_fn(row, pred, my_prob, bankroll)
            if size > bankroll: size = bankroll
            
            if should_bet and size > 0:
                bets_placed += 1
                wagered += size
                if won_bet:
                    payout = size * odds
                    returned += payout
                    bankroll += (payout - size)
                    wins += 1
                else:
                    bankroll -= size
        
        net_profit = returned - wagered
        roi = (net_profit / wagered * 100) if wagered > 0 else 0.0
        
        results[name] = {
            "roi": roi,
            "final_bankroll": bankroll,
            "bets": bets_placed
        }
    return results

def is_value_bet(row, pred, prob, margin):
    if pred == 1: odds = row['f_1_odds']
    else: odds = row['f_2_odds']
    if odds <= 1.0: return False, 0.0
    implied = 1.0 / odds
    if (prob - implied) > margin: return True, 10.0
    return False, 0.0

def kelly_bet(row, pred, prob, bankroll, fraction=1.0):
    if pred == 1: odds = row['f_1_odds']
    else: odds = row['f_2_odds']
    if odds <= 1.0: return False, 0.0
    b = odds - 1.0
    q = 1.0 - prob
    f_star = (b * prob - q) / b
    if f_star > 0:
        stake = bankroll * f_star * fraction
        max_stake = bankroll * 0.20
        if stake > max_stake: stake = max_stake
        return True, stake
    return False, 0.0

# --- Main Simulation Loop ---
def run_monte_carlo(iterations=10):
    print(f"=== Starting Monte Carlo Simulation ({iterations} runs) ===")
    
    # Load Data Once
    if not os.path.exists('data/training_data.csv'): return
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    with open('features.json', 'r') as f: features = json.load(f)
    with open('params.json', 'r') as f: params = json.load(f)['best_params']
    
    # Split
    split_date = '2025-01-01'
    train_df = df[df['event_date'] < split_date].copy()
    test_df_raw = df[df['event_date'] >= split_date].copy()
    test_df = test_df_raw[(test_df_raw['f_1_odds'] > 1.0) & (test_df_raw['f_2_odds'] > 1.0)].copy()
    
    print(f"Train: {len(train_df)}, Test (Valid Odds): {len(test_df)}")
    
    # Prepare Data
    X_train = train_df[[c for c in features if c in train_df.columns]].fillna(0)
    y_train = train_df['target'].values
    X_test = test_df[[c for c in features if c in test_df.columns]].fillna(0)
    y_test = test_df['target'].values
    
    # Siamese Prep (Static part)
    f1_train, f2_train, input_dim, _ = prepare_siamese_data(X_train, features)
    f1_test, f2_test, _, _ = prepare_siamese_data(X_test, features)
    scaler = StandardScaler()
    combined_train = np.concatenate([f1_train, f2_train], axis=0)
    scaler.fit(combined_train)
    f1_train_s = scaler.transform(f1_train)
    f2_train_s = scaler.transform(f2_train)
    f1_test_s = scaler.transform(f1_test)
    f2_test_s = scaler.transform(f2_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Storage for results
    agg_results = {k: {'roi': [], 'bankroll': []} for k in [
        "Flat (All Bets)", "Value (Edge > 5%)", "Kelly (Full)", "Kelly (1/4)", "Kelly (1/8)"
    ]}
    
    for i in range(iterations):
        print(f"Run {i+1}/{iterations}...", end='\r')
        
        # 1. Train XGBoost (Deterministic if seed set, but let's vary seed slightly or keep fixed?)
        # User wants to see variance. Variance comes from Siamese mostly.
        # Let's use a different random state for each run to simulate "training noise"
        current_seed = 42 + i
        
        xgb_model = xgb.XGBClassifier(
            max_depth=params['xgb_max_depth'],
            learning_rate=params['xgb_learning_rate'],
            n_estimators=params['xgb_n_estimators'],
            min_child_weight=params['xgb_min_child_weight'],
            subsample=params['xgb_subsample'],
            colsample_bytree=params['xgb_colsample_bytree'],
            eval_metric='logloss',
            random_state=current_seed,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        
        # 2. Train Siamese (Stochastic)
        torch.manual_seed(current_seed) # Seed torch
        siamese_model = SiameseMatchupNet(input_dim, hidden_dim=params['siamese_hidden_dim']).to(device)
        train_ds = TensorDataset(torch.FloatTensor(f1_train_s), torch.FloatTensor(f2_train_s), torch.FloatTensor(y_train))
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
            t_f1 = torch.FloatTensor(f1_test_s).to(device)
            t_f2 = torch.FloatTensor(f2_test_s).to(device)
            siamese_probs = siamese_model(t_f1, t_f2).cpu().numpy()
            
        # 3. Ensemble
        w = params['ensemble_xgb_weight']
        ens_probs = w * xgb_probs + (1 - w) * siamese_probs
        ens_preds = (ens_probs > 0.5).astype(int)
        
        # 4. Evaluate
        run_res = evaluate_strategies(test_df, ens_preds, ens_probs)
        for k, v in run_res.items():
            agg_results[k]['roi'].append(v['roi'])
            agg_results[k]['bankroll'].append(v['final_bankroll'])
            
    print("\n\n=== Monte Carlo Results (10 Runs) ===")
    print(f"{'Strategy':<20} | {'Mean ROI':<10} | {'Std Dev':<10} | {'Mean Bankroll':<15} | {'Min Bankroll':<15} | {'Max Bankroll':<15}")
    print("-" * 100)
    
    for name, data in agg_results.items():
        mean_roi = np.mean(data['roi'])
        std_roi = np.std(data['roi'])
        mean_bk = np.mean(data['bankroll'])
        min_bk = np.min(data['bankroll'])
        max_bk = np.max(data['bankroll'])
        
        print(f"{name:<20} | {mean_roi:>9.2f}% | {std_roi:>9.2f}% | ${mean_bk:>14.2f} | ${min_bk:>14.2f} | ${max_bk:>14.2f}")

if __name__ == "__main__":
    run_monte_carlo(10)
