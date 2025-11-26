import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import json
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from experimental.decision.profit_objectives import simulate_betting

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Check for odds columns
    # Assuming columns like 'f_1_odds', 'f_2_odds' exist or similar.
    # If not, we might need to mock them or use implied probs if available.
    
    # Let's check columns first.
    # Based on previous context, we have 'f_1_implied_prob' etc.
    # Odds = 1 / Implied Prob (roughly)
    
    if 'f_1_odds' not in df.columns:
        if 'f_1_implied_prob' in df.columns:
            print("Calculating odds from implied probabilities...")
            df['f_1_odds'] = 1.0 / df['f_1_implied_prob']
            df['f_2_odds'] = 1.0 / df['f_2_implied_prob']
        else:
            print("Error: No odds or implied probability columns found.")
            return

    # Load features
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    # Encode categoricals
    cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    # Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    # Split
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    
    X_test = test_df[features]
    y_test = test_df['target']
    
    odds_f1_test = test_df['f_1_odds'].values
    odds_f2_test = test_df['f_2_odds'].values
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # 1. Baseline Model (Standard Log Loss)
    print("\nTraining Baseline Model...")
    xgb_base = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1
    )
    xgb_base.fit(X_train, y_train)
    
    prob_base = xgb_base.predict_proba(X_test)[:, 1]
    acc_base = accuracy_score(y_test, (prob_base > 0.5).astype(int))
    ll_base = log_loss(y_test, prob_base)
    print(f"Baseline: Acc {acc_base:.4f}, LL {ll_base:.4f}")
    
    # 2. Profit-Weighted Model
    # Weight = log(Odds of Winner)
    # If F1 wins, weight = log(Odds_F1)
    # If F2 wins, weight = log(Odds_F2)
    # Idea: We care more about predicting correctly when the payout is high.
    
    print("\nTraining Profit-Weighted Model...")
    
    # Calculate sample weights for training
    w_train = []
    for idx, row in train_df.iterrows():
        if row['target'] == 1:
            w = np.log(row['f_1_odds']) if row['f_1_odds'] > 1 else 0.1
        else:
            w = np.log(row['f_2_odds']) if row['f_2_odds'] > 1 else 0.1
        w_train.append(w)
    
    w_train = np.array(w_train)
    # Normalize weights
    w_train = w_train / w_train.mean()
    
    xgb_profit = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1
    )
    xgb_profit.fit(X_train, y_train, sample_weight=w_train)
    
    prob_profit = xgb_profit.predict_proba(X_test)[:, 1]
    acc_profit = accuracy_score(y_test, (prob_profit > 0.5).astype(int))
    ll_profit = log_loss(y_test, prob_profit)
    print(f"Profit-Weighted: Acc {acc_profit:.4f}, LL {ll_profit:.4f}")
    
    # 3. Simulate Betting
    print("\n--- Betting Simulation (Initial Bankroll $1000) ---")
    
    results = []
    
    # Baseline Flat
    end_bf, roi_bf, _ = simulate_betting(prob_base, y_test.values, odds_f1_test, odds_f2_test, strategy='flat')
    results.append({'Model': 'Baseline', 'Strategy': 'Flat (5%)', 'Bankroll': end_bf, 'ROI': roi_bf})
    
    # Baseline Kelly
    end_bk, roi_bk, _ = simulate_betting(prob_base, y_test.values, odds_f1_test, odds_f2_test, strategy='kelly')
    results.append({'Model': 'Baseline', 'Strategy': 'Kelly (1/4)', 'Bankroll': end_bk, 'ROI': roi_bk})
    
    # Profit Flat
    end_pf, roi_pf, _ = simulate_betting(prob_profit, y_test.values, odds_f1_test, odds_f2_test, strategy='flat')
    results.append({'Model': 'Profit-Weighted', 'Strategy': 'Flat (5%)', 'Bankroll': end_pf, 'ROI': roi_pf})
    
    # Profit Kelly
    end_pk, roi_pk, _ = simulate_betting(prob_profit, y_test.values, odds_f1_test, odds_f2_test, strategy='kelly')
    results.append({'Model': 'Profit-Weighted', 'Strategy': 'Kelly (1/4)', 'Bankroll': end_pk, 'ROI': roi_pk})
    
    # Display Results
    res_df = pd.DataFrame(results)
    print(res_df)
    
    # Save results
    with open('experimental/PROFIT_RESULTS.md', 'w') as f:
        f.write(f"# Profit-Aware Loss & Staking Results\n\n")
        f.write(f"## Model Metrics\n")
        f.write(f"| Model | Accuracy | Log Loss |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| Baseline | {acc_base:.4f} | {ll_base:.4f} |\n")
        f.write(f"| Profit-Weighted | {acc_profit:.4f} | {ll_profit:.4f} |\n\n")
        
        f.write(f"## Betting Simulation (Holdout Set)\n")
        f.write(res_df.to_markdown(index=False))
        
        f.write(f"\n\n## Interpretation\n")
        best_roi = res_df['ROI'].max()
        best_strat = res_df.loc[res_df['ROI'].idxmax()]
        f.write(f"Best Strategy: **{best_strat['Model']} + {best_strat['Strategy']}** with ROI **{best_roi*100:.2f}%**.\n")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
