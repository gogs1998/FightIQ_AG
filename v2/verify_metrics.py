import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.analyst_ensemble import AnalystEnsemble
from models.gambler_model import GamblerModel

def verify_metrics():
    print("Loading Data...")
    df = pd.read_csv('v2/data/training_data_v2.csv')
    
    # Split (Last 15% as Holdout/Test)
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy()
    y_test = test_df['target']
    
    print(f"Test Set Size: {len(test_df)}")
    
    # 1. Evaluate Analyst Model
    print("\n--- Analyst Model (The Truth) ---", flush=True)
    try:
        analyst = AnalystEnsemble.load('v2/models/analyst_model.pkl')
        probs, sets = analyst.predict(test_df)
        
        acc = accuracy_score(y_test, (probs > 0.5).astype(int))
        ll = log_loss(y_test, probs)
        
        # Conformal Stats
        # sets is a boolean array [n_samples, n_classes]
        # y_test is the true class index (0 or 1)
        
        # Reset index of y_test to match sets (which is 0..N-1)
        y_true = y_test.values
        
        print(f"Sets shape: {sets.shape}", flush=True)
        print(f"y_true shape: {y_true.shape}", flush=True)
        print(f"y_true unique values: {np.unique(y_true)}", flush=True)
        
        n_singleton = np.sum([1 if np.sum(s) == 1 else 0 for s in sets])
        coverage = np.mean([1 if s[y] else 0 for s, y in zip(sets, y_true)])
        
        print(f"Accuracy: {acc:.4%}", flush=True)
        print(f"Log Loss: {ll:.4f}", flush=True)
        print(f"Conformal Coverage: {coverage:.4%}", flush=True)
        print(f"Singleton (Certain) Rate: {n_singleton / len(test_df):.4%}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error evaluating Analyst: {e}", flush=True)

    # 2. Evaluate Gambler Model
    print("\n--- Gambler Model (The Money) ---")
    try:
        gambler = GamblerModel.load('v2/models/gambler_model.pkl')
        g_probs = gambler.predict(test_df)
        
        # ROI Calculation
        # We need odds. Assuming they are in the df as 'f_1_odds', 'f_2_odds'
        # If not, we can't calc ROI easily.
        # Let's check columns
        if 'f_1_odds' in test_df.columns:
            bets = gambler.recommend_bets(test_df, g_probs, bankroll=100)
            total_wagered = sum(b['wager'] for b in bets)
            total_return = 0
            
            for idx, (bet, row) in enumerate(zip(bets, test_df.iterrows())):
                row = row[1]
                winner = row['winner']
                # Bet is on 'fighter'
                if bet['wager'] > 0:
                    if bet['fighter'] == winner:
                        # Profit = Wager * (Odds - 1) + Wager = Wager * Odds
                        total_return += bet['wager'] * bet['odds']
            
            profit = total_return - total_wagered
            roi = profit / total_wagered if total_wagered > 0 else 0
            
            print(f"Total Wagered: ${total_wagered:.2f}")
            print(f"Total Profit: ${profit:.2f}")
            print(f"ROI: {roi:.4%}")
        else:
            print("Odds columns not found, skipping ROI.")
            
    except Exception as e:
        print(f"Error evaluating Gambler: {e}")

if __name__ == "__main__":
    verify_metrics()
