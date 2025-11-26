import pandas as pd
import joblib
import json
import numpy as np

def simulate_aggressive():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
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
    
    # Load Model
    model = joblib.load('ufc_model_elo.pkl')
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    X_test = test_df[features]
    
    # Predict
    probs = model.predict_proba(X_test)[:, 1]
    test_df['prob_f1'] = probs
    test_df['pred_winner'] = np.where(probs > 0.5, 1, 0)
    test_df['confidence'] = np.where(test_df['pred_winner'] == 1, test_df['prob_f1'], 1 - test_df['prob_f1'])
    test_df['correct'] = test_df['pred_winner'] == test_df['target']
    
    # Filter for High Confidence (> 80%)
    high_conf = test_df[test_df['confidence'] >= 0.8].copy()
    
    print(f"\nTotal Test Fights: {len(test_df)}")
    print(f"High Confidence Bets (>= 80%): {len(high_conf)}")
    
    # Betting Simulation
    initial_bankroll = 1000
    bankroll = initial_bankroll
    bet_amount = 100
    
    roi_data = []
    wins = 0
    losses = 0
    
    print("\n--- Aggressive Betting Simulation (Confidence >= 80%) ---")
    
    for idx, row in high_conf.iterrows():
        # Check if odds exist
        o1 = row['f_1_odds']
        o2 = row['f_2_odds']
        
        if pd.isna(o1) or pd.isna(o2):
            continue
            
        # Determine bet
        # We bet on the predicted winner
        if row['pred_winner'] == 1:
            odds = o1
            won = row['target'] == 1
        else:
            odds = o2
            won = row['target'] == 0
            
        if won:
            profit = bet_amount * (odds - 1)
            bankroll += profit
            roi_data.append(profit)
            wins += 1
        else:
            bankroll -= bet_amount
            roi_data.append(-bet_amount)
            losses += 1
            
    print(f"Bets Placed: {len(roi_data)}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {wins/len(roi_data):.2%}")
    
    profit_loss = bankroll - initial_bankroll
    roi = (sum(roi_data) / (len(roi_data) * bet_amount)) * 100
    
    print(f"\nInitial Bankroll: ${initial_bankroll}")
    print(f"Final Bankroll: ${bankroll:.2f}")
    print(f"Profit/Loss: ${profit_loss:.2f}")
    print(f"ROI: {roi:.2f}%")

if __name__ == "__main__":
    simulate_aggressive()
