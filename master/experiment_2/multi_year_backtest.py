import pandas as pd
import numpy as np
import xgboost as xgb
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss

def run_multi_year_backtest():
    print("=== FightIQ: Multi-Year Robustness Test (2021-2025) ===")
    
    # 1. Load Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 2. Load Features & Params
    with open(f'{BASE_DIR}/prop_hunter/features.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
        
    # 3. Define Split
    # Train: 2010 - 2020
    # Test Years: 2021, 2022, 2023, 2024, 2025
    
    train_mask = (df['event_date'] >= '2010-01-01') & (df['event_date'] < '2021-01-01')
    train_df = df[train_mask]
    
    X_train = train_df[features].fillna(0)
    y_train = train_df['target']
    
    print(f"Training on 2010-2020 Data ({len(train_df)} fights)...")
    
    # Train Model ONCE (Simulating a model built in Jan 2021)
    # Ideally we should retrain annually (Walk-Forward), but let's test the "decay" of a static model first.
    # Actually, user asked to "train up to 2020", implying a static split.
    # But for a realistic backtest, we should probably retrain or at least see how it holds up.
    # Let's do STATIC first to see if the "Alpha" decays.
    
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate Each Year
    years = [2021, 2022, 2023, 2024, 2025]
    results = []
    
    print("\n=== Annual Performance (Static Model trained on <2021) ===")
    print(f"{'Year':<6} | {'Fights':<6} | {'Acc':<6} | {'LogLoss':<7} | {'ROI (Flat)':<10} | {'ROI (Sniper)':<10}")
    print("-" * 65)
    
    total_profit_sniper = 0
    total_invested_sniper = 0
    
    for year in years:
        mask_year = (df['event_date'] >= f'{year}-01-01') & (df['event_date'] < f'{year+1}-01-01')
        test_df = df[mask_year].copy()
        
        if len(test_df) == 0: continue
        
        X_test = test_df[features].fillna(0)
        y_test = test_df['target']
        
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, probs)
        
        # Betting Simulation
        profit_flat = 0
        profit_sniper = 0
        invested_sniper = 0
        
        for i, idx in enumerate(test_df.index):
            row = test_df.loc[idx]
            prob = probs[i]
            target = y_test.iloc[i]
            
            # Odds
            odds_1 = row['f_1_odds']
            odds_2 = row['f_2_odds']
            
            if pd.isna(odds_1) or pd.isna(odds_2): continue
            
            # Flat Betting (on Winner)
            if prob > 0.5:
                bet_on = 1
                odds = odds_1
            else:
                bet_on = 0
                odds = odds_2
                
            if bet_on == target:
                profit_flat += (odds - 1)
            else:
                profit_flat -= 1
                
            # Value Sniper (>5% Edge)
            edge = 0
            if prob > 0.5:
                implied = 1/odds_1
                if (prob - implied) > 0.05:
                    bet_on_sniper = 1
                    odds_sniper = odds_1
                    edge = prob - implied
            else:
                implied = 1/odds_2
                if ((1-prob) - implied) > 0.05:
                    bet_on_sniper = 0
                    odds_sniper = odds_2
                    edge = (1-prob) - implied
            
            if edge > 0:
                invested_sniper += 1
                if bet_on_sniper == target:
                    profit_sniper += (odds_sniper - 1)
                else:
                    profit_sniper -= 1
                    
        roi_flat = profit_flat / len(test_df) if len(test_df) > 0 else 0
        roi_sniper = profit_sniper / invested_sniper if invested_sniper > 0 else 0
        
        total_profit_sniper += profit_sniper
        total_invested_sniper += invested_sniper
        
        print(f"{year:<6} | {len(test_df):<6} | {acc:<6.1%} | {ll:<7.4f} | {roi_flat:<10.1%} | {roi_sniper:<10.1%}")
        
        results.append({
            "year": year,
            "acc": acc,
            "roi_sniper": roi_sniper
        })
        
    print("-" * 65)
    print(f"Total Sniper ROI (2021-2025): {total_profit_sniper/total_invested_sniper:.1%}")
    
    # 5. Genetic Optimization (Quick Check)
    # We want to see if the "Optimal Strategy" changes if we optimize on 2010-2020 vs 2010-2023.
    # This is a complex task, but we can simulate it by checking if the "Sniper" threshold (5%) holds up.
    
    # Plot ROI by Year
    years_plot = [r['year'] for r in results]
    rois = [r['roi_sniper'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(years_plot, rois, color=['green' if x > 0 else 'red' for x in rois])
    plt.title("Value Sniper Strategy ROI (2021-2025)", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("ROI")
    plt.axhline(0, color='black', linestyle='--')
    plt.savefig(f'{BASE_DIR}/experiment_2/viz_multi_year_roi.png')
    print("\nSaved viz_multi_year_roi.png")

if __name__ == "__main__":
    run_multi_year_backtest()
