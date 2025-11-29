import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
import matplotlib.pyplot as plt

def backtest_optimized_boruta():
    print("=== FightIQ Optimized Boruta Backtest (2024-2025) ===")
    
    # 1. Load Data & Features
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    with open(f'{BASE_DIR}/experiment_2/boruta_results.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
        
    print(f"Using {len(features)} features and Optimized Hyperparameters.")
    print(f"Params: {params}")
        
    # Filter valid odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    split_date = '2024-01-01'
    mask_train = df['event_date'] < split_date
    mask_test = df['event_date'] >= split_date
    
    X_train = X[mask_train]
    X_test = X[mask_test]
    y_train = y[mask_train]
    y_test = y[mask_test]
    
    print("Training Optimized Model...")
    # Add fixed params that weren't optimized but needed
    params['eval_metric'] = 'logloss'
    params['n_jobs'] = -1
    params['random_state'] = 42
    params['use_label_encoder'] = False
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    
    # 2. Prepare Backtest DataFrame
    test_df = df[mask_test].copy()
    test_df['prob'] = probs
    test_df['target'] = y_test
    
    print(f"Backtesting on {len(test_df)} fights...")
    
    # 3. Define Strategies
    strategies = {
        "Flat (Conf > 60%)": [],
        "Value Sniper (Edge > 5%)": [],
        "Kelly (Eighth, Conf > 55%)": []
    }
    
    bankrolls = {k: 1000.0 for k in strategies}
    history = {k: [1000.0] for k in strategies}
    
    for idx, row in test_df.iterrows():
        p = row['prob']
        
        # Determine Bet
        if p > 0.5:
            my_prob = p
            odds = row['f_1_odds']
            win = (row['target'] == 1)
        else:
            my_prob = 1 - p
            odds = row['f_2_odds']
            win = (row['target'] == 0)
            
        edge = my_prob - (1/odds)
        
        # --- Strategy Logic ---
        
        # 1. Flat > 60%
        if my_prob > 0.60:
            stake = 50 # 5% unit
            res = (stake * (odds - 1)) if win else -stake
            bankrolls["Flat (Conf > 60%)"] += res
            
        # 2. Value Sniper
        if edge > 0.05:
            stake = 50
            res = (stake * (odds - 1)) if win else -stake
            bankrolls["Value Sniper (Edge > 5%)"] += res
            
        # 3. Kelly Eighth
        if my_prob > 0.55 and edge > 0:
            b = odds - 1
            q = 1 - my_prob
            f = (b * my_prob - q) / b
            stake = bankrolls["Kelly (Eighth, Conf > 55%)"] * (f * 0.125)
            if stake < 0: stake = 0
            res = (stake * (odds - 1)) if win else -stake
            bankrolls["Kelly (Eighth, Conf > 55%)"] += res
            
        # Update History
        for k in strategies:
            history[k].append(bankrolls[k])
            
    # 4. Report
    print("\n=== Final Results (Start $1,000) ===")
    print(f"{'Strategy':<40} | {'End Bank':<12} | {'ROI':<8} | {'Max DD':<8}")
    print("-" * 80)
    
    for name, bank in bankrolls.items():
        hist = history[name]
        roi = (bank - 1000) / 1000
        
        # Calc Max Drawdown
        peak = 1000
        max_dd = 0
        for val in hist:
            if val > peak: peak = val
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
            
        print(f"{name:<40} | ${bank:,.2f}   | {roi:<8.1%} | {max_dd:<8.1%}")
        
    # Plot
    plt.figure(figsize=(12, 6))
    for name, hist in history.items():
        plt.plot(hist, label=name)
    plt.title("FightIQ Strategy Backtest (2024-2025) - Optimized Boruta")
    plt.xlabel("Bets Placed")
    plt.ylabel("Bankroll ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{BASE_DIR}/experiment_2/strategy_backtest_optimized.png')
    print("\nSaved plot to experiment_2/strategy_backtest_optimized.png")

if __name__ == "__main__":
    backtest_optimized_boruta()
