import pandas as pd
import joblib
import json
import numpy as np

def simulate_betting():
    print("Loading data...")
    df = pd.read_csv('UFC_full_data_golden.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Create Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    # Split (Test set only)
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Test Set Size: {len(test_df)}")
    
    # Load Model
    model = joblib.load('ufc_model_optimized.pkl')
    with open('final_features.json', 'r') as f:
        features = json.load(f)
        
    X_test = test_df[features]
    y_test = test_df['target']
    
    # Predict Probabilities
    probs = model.predict_proba(X_test)[:, 1]
    test_df['prob_f1'] = probs
    test_df['pred_winner'] = np.where(probs > 0.5, 1, 0)
    test_df['correct'] = test_df['pred_winner'] == test_df['target']
    
    # Betting Simulation
    initial_bankroll = 1000
    bankroll = initial_bankroll
    history = []
    
    # Kelly Criterion or Flat Bet? Let's try Flat Bet first.
    bet_amount = 100
    
    print("\n--- Betting Simulation (Flat $100) ---")
    
    roi_data = []
    
    for idx, row in test_df.iterrows():
        # Odds are Decimal
        o1 = row['f_1_odds']
        o2 = row['f_2_odds']
        
        if pd.isna(o1) or pd.isna(o2):
            continue
            
        # Determine who we bet on
        # Simple strategy: Bet on the predicted winner if there is "value"
        # Value = Probability > 1/Odds
        
        p_win = row['prob_f1']
        if p_win > 0.5:
            # Bet on F1
            implied_prob = 1 / o1
            edge = p_win - implied_prob
            
            if edge > 0.05: # 5% edge threshold
                wager = bet_amount
                if row['target'] == 1:
                    profit = wager * (o1 - 1)
                    bankroll += profit
                    roi_data.append(profit)
                else:
                    bankroll -= wager
                    roi_data.append(-wager)
        else:
            # Bet on F2
            p_win_f2 = 1 - p_win
            implied_prob = 1 / o2
            edge = p_win_f2 - implied_prob
            
            if edge > 0.05:
                wager = bet_amount
                if row['target'] == 0:
                    profit = wager * (o2 - 1)
                    bankroll += profit
                    roi_data.append(profit)
                else:
                    bankroll -= wager
                    roi_data.append(-wager)
                    
    print(f"Initial Bankroll: ${initial_bankroll}")
    print(f"Final Bankroll: ${bankroll:.2f}")
    print(f"Total Bets: {len(roi_data)}")
    print(f"Profit/Loss: ${bankroll - initial_bankroll:.2f}")
    if len(roi_data) > 0:
        roi = (sum(roi_data) / (len(roi_data) * bet_amount)) * 100
        print(f"ROI: {roi:.2f}%")

if __name__ == "__main__":
    simulate_betting()
