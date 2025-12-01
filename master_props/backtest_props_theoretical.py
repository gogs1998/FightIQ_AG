import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
import matplotlib.pyplot as plt

def backtest_props_theoretical():
    print("=== Prop Hunter: Theoretical Backtest (Conservative Odds) ===")
    
    # 1. Load Data & Models
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    with open(f'{BASE_DIR}/prop_hunter/features.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    model_finish = joblib.load(f'{BASE_DIR}/prop_hunter/model_finish.pkl')
    model_method = joblib.load(f'{BASE_DIR}/prop_hunter/model_method.pkl')
    
    # Filter valid odds & time
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Test Set (2024-2025)
    mask_test = df['event_date'] >= '2024-01-01'
    test_df = df[mask_test].copy()
    
    X = test_df[[c for c in features if c in test_df.columns]].fillna(0)
    
    print(f"Backtesting on {len(test_df)} fights...")
    
    # 2. Generate Predictions
    p_finish = model_finish.predict_proba(X)[:, 1]
    p_ko = model_method.predict_proba(X)[:, 1]
    
    # 3. Define Conservative Odds Assumptions
    # These are "worst case" market odds for these props.
    # If we profit here, we profit everywhere.
    ODDS_KO = 2.00      # Even money (Very conservative for KO props)
    ODDS_SUB = 3.50     # +250 (Conservative for Sub props)
    ODDS_DEC = 1.60     # -167 (Standard for GTD)
    ODDS_FINISH = 1.60  # -167 (Standard for ITD)
    
    bankroll = 1000.0
    history = [1000.0]
    stake = 50.0 # Flat bet
    
    bets_placed = 0
    wins = 0
    
    print(f"\nAssumed Odds: KO={ODDS_KO}, Sub={ODDS_SUB}, Dec={ODDS_DEC}, Finish={ODDS_FINISH}")
    print("-" * 60)
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # Probs
        pf = p_finish[i]
        p_dec = 1 - pf
        pk = p_ko[i] # P(KO|Finish)
        ps = 1 - pk  # P(Sub|Finish)
        
        # Combined Probs
        prob_ko = pf * pk
        prob_sub = pf * ps
        
        # Strategy Rules (from Strategy.md)
        # 1. KO: P(KO) > 0.50
        if prob_ko > 0.50:
            bets_placed += 1
            res = str(row['result']).lower()
            won = ('ko' in res or 'tko' in res)
            
            profit = (stake * (ODDS_KO - 1)) if won else -stake
            bankroll += profit
            if won: wins += 1
            
        # 2. Sub: P(Sub) > 0.30
        elif prob_sub > 0.30: # Elif to avoid double betting on same fight? Or allow multiple?
            # Strategy usually allows multiple if edge exists, but let's be safe and separate.
            # Actually, KO and Sub are mutually exclusive, so we can check both.
            bets_placed += 1
            res = str(row['result']).lower()
            won = ('submission' in res)
            
            profit = (stake * (ODDS_SUB - 1)) if won else -stake
            bankroll += profit
            if won: wins += 1
            
        # 3. Decision: P(Dec) > 0.60
        # Note: Decision is mutually exclusive with KO/Sub.
        elif p_dec > 0.60:
            bets_placed += 1
            res = str(row['result']).lower()
            won = ('decision' in res)
            
            profit = (stake * (ODDS_DEC - 1)) if won else -stake
            bankroll += profit
            if won: wins += 1
            
        history.append(bankroll)
        
    # 4. Results
    roi = (bankroll - 1000) / 1000
    win_rate = wins / bets_placed if bets_placed > 0 else 0
    
    print(f"\n=== Theoretical Results (2024-2025) ===")
    print(f"Final Bankroll: ${bankroll:,.2f}")
    print(f"Total Profit:   ${bankroll - 1000:,.2f}")
    print(f"ROI:            {roi:.2%}")
    print(f"Total Bets:     {bets_placed}")
    print(f"Win Rate:       {win_rate:.2%}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title("Prop Hunter Theoretical Backtest (Conservative Odds)")
    plt.xlabel("Bets")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    plt.savefig(f'{BASE_DIR}/prop_hunter/theoretical_backtest.png')
    print("\nSaved plot to prop_hunter/theoretical_backtest.png")

if __name__ == "__main__":
    backtest_props_theoretical()
