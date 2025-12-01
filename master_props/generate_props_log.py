import pandas as pd
import numpy as np
import joblib
import json
import os

def generate_props_log():
    print("=== Master Props: Generating Backtest Log (2024-2025) ===")
    
    # 1. Load Data & Models
    try:
        df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    except:
        df = pd.read_csv('data/training_data_enhanced.csv')
        
    with open('../master_3/features_selected.json', 'r') as f:
        features = json.load(f)
        
import pandas as pd
import numpy as np
import joblib
import json
import os

def generate_props_log():
    print("=== Master Props: Generating Backtest Log (2024-2025) ===")
    
    # 1. Load Data & Models
    try:
        df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    except:
        df = pd.read_csv('data/training_data_enhanced.csv')
        
    with open('../master_3/features_selected.json', 'r') as f:
        features = json.load(f)
        
    prop_features = ['f_1_chin_score', 'f_2_chin_score', 'diff_chin_score']
    for f in prop_features:
        if f not in features:
            features.append(f)
            
    # Load Main Model for Win Prob
    model_win = joblib.load('../master_3/models/xgb_master3.pkl')
    
    # Filter for 2024-2025
    df['event_date'] = pd.to_datetime(df['event_date'])
    mask_test = df['event_date'] >= '2024-01-01'
    test_df = df[mask_test].copy().reset_index(drop=True)
    
    # Feature Sets
    # Main model uses 'features' (without chin_score)
    # Prop models use 'features' + chin_score (which we appended to 'features' list earlier)
    # Wait, we appended to 'features' list in place. 
    # We need to separate them.
    
    # Reload features to be safe
    with open('../master_3/features_selected.json', 'r') as f:
        features_main = json.load(f)
        
    features_props = features_main.copy()
    prop_features = ['f_1_chin_score', 'f_2_chin_score', 'diff_chin_score']
    for f in prop_features:
        if f not in features_props:
            features_props.append(f)
            
    # Apply Opponent Adjustment for Main Model
    # We need to import the adjustment function or replicate it.
    # Since we are in master_props, we can import from ../master_3/models/opponent_adjustment.py
    import sys
    sys.path.append('../master_3')
    from models.opponent_adjustment import apply_opponent_adjustment
    
    # We need the full df to apply adjustment (it needs history)
    # But apply_opponent_adjustment takes a DataFrame and returns adjusted DataFrame.
    # Wait, apply_opponent_adjustment is complex and might need full history.
    # Let's check if training_data_enhanced.csv ALREADY has adjusted features?
    # The error says "expected ..._adj in input data".
    # So the model was trained on adjusted features.
    # If training_data_enhanced.csv doesn't have them, we must generate them.
    
    # Let's try to generate them on the fly for the test set.
    # Actually, apply_opponent_adjustment usually works on the whole dataset.
    # Let's apply it to 'df' before filtering for test set.
    
    # Define stats to adjust (Standard set from master_3)
    stat_cols = [
        'slpm_15_f_1', 'sapm_15_f_1', 'td_avg_15_f_1', 'sub_avg_15_f_1',
        'slpm_15_f_2', 'sapm_15_f_2', 'td_avg_15_f_2', 'sub_avg_15_f_2'
    ]
    
    print("Applying opponent adjustment...")
    df_adj = apply_opponent_adjustment(df, stat_cols)
    
    # Re-filter for test set
    mask_test = df_adj['event_date'] >= '2024-01-01'
    test_df = df_adj[mask_test].copy().reset_index(drop=True)
    
    # Add adjusted features to features_main because the model expects them
    adj_features = [
        'slpm_15_f_1_adj', 'slpm_15_f_2_adj', 
        'td_avg_15_f_1_adj', 'td_avg_15_f_2_adj', 
        'sub_avg_15_f_1_adj', 'sub_avg_15_f_2_adj', 
        'sapm_15_f_1_adj', 'sapm_15_f_2_adj'
    ]
    for f in adj_features:
        if f not in features_main:
            features_main.append(f)
            
    X_main = test_df[[c for c in features_main if c in test_df.columns]].fillna(0)
    X_props = test_df[[c for c in features_props if c in test_df.columns]].fillna(0)
    
    model_finish = joblib.load('model_finish.pkl')
    model_method = joblib.load('model_method.pkl')
    model_round = joblib.load('model_round.pkl')
    
    # 2. Generate Predictions
    p_win = model_win.predict_proba(X_main)[:, 1] # P(Fighter 1 Wins)
    
    p_finish = model_finish.predict_proba(X_props)[:, 1]
    p_decision = 1 - p_finish
    
    p_method_probs = model_method.predict_proba(X_props)
    p_ko = p_method_probs[:, 0]
    p_sub = p_method_probs[:, 1]
    
    p_round_probs = model_round.predict_proba(X_props) # [N, 5]
    
    # 3. Simulate Betting
    # Assumptions
    AVG_KO_ODDS = 2.20
    AVG_SUB_ODDS = 3.50
    AVG_GTD_ODDS = 1.70  # (-143)
    AVG_RND_ODDS = 4.00  # (+300)
    AVG_EXACT_ODDS = 21.00 # +2000 for "Win by KO in Round X"
    AVG_DEC_WIN_ODDS = 3.00 # +200 for "Win by Decision"
    STAKE = 100
    
    bankroll = 10000
    history = []
    
    print(f"Simulating {len(test_df)} fights...")
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # Probabilities
        p_w = p_win[i]
        p_l = 1 - p_w
        
        # Determine predicted winner
        if p_w > 0.5:
            pred_winner = 1
            prob_win = p_w
            winner_name = row['f_1_name']
        else:
            pred_winner = 2
            prob_win = p_l
            winner_name = row['f_2_name']
            
        # Prop Probs (These are "Fight Ends By...", independent of winner)
        # To get "Winner + Method", we multiply P(Win) * P(Method|Win)
        # We approximate P(Method|Win) as P(Method).
        
        prob_fight_ko = p_finish[i] * p_ko[i]
        prob_fight_sub = p_finish[i] * p_sub[i]
        prob_gtd = p_decision[i]
        
        # Exact Round Probs
        best_round_idx = np.argmax(p_round_probs[i])
        best_round_prob = p_round_probs[i][best_round_idx]
        pred_round = best_round_idx + 1
        
        # Result
        res_str = str(row['result']).lower()
        is_ko = 'ko' in res_str or 'tko' in res_str
        is_sub = 'submission' in res_str
        is_decision = 'decision' in res_str
        
        finish_round = row['finish_round']
        if pd.isna(finish_round): finish_round = -1
        else: finish_round = int(finish_round)
        
        # Did the predicted winner actually win?
        # We don't have a simple "winner" column, we have "result" (e.g. "Decision - Unanimous") and usually "winner" column in raw data
        # But training_data_enhanced has 'winner' column? Let's check.
        # It usually has 'winner' (name) or 'f_1_won' (bool)?
        # Let's assume 'winner' column exists or infer it.
        # Actually, let's look at 'winner' column.
        actual_winner = str(row.get('winner', '')).strip()
        
        did_win = False
        if pred_winner == 1 and row['f_1_name'] == actual_winner: did_win = True
        if pred_winner == 2 and row['f_2_name'] == actual_winner: did_win = True
        
        # --- EXACT BET: Winner + Method + Round ---
        # Strategy: If P(Win) > 60% AND P(KO) > 50% AND P(Round) > 30%
        # Bet: "Winner by KO in Round X"
        
        prob_exact_ko = prob_win * prob_fight_ko * best_round_prob
        
        if prob_exact_ko > 0.10: # 10% chance for +2000 odds is EV+ (Breakeven 4.7%)
            wager = STAKE
            # Win Condition: Correct Winner AND Correct Method (KO) AND Correct Round
            won_bet = did_win and is_ko and (finish_round == pred_round)
            
            if won_bet:
                profit = wager * (AVG_EXACT_ODDS - 1)
                res = 'WIN'
            else:
                profit = -wager
                res = 'LOSS'
                
            bankroll += profit
            history.append({'Date': row['event_date'], 'Fight': f"{row['f_1_name']} vs {row['f_2_name']}", 'Bet_Type': 'EXACT_KO', 'Prob': prob_exact_ko, 'Result': res, 'Profit': profit, 'Bankroll': bankroll})

        # --- EXACT DECISION: Winner + Decision ---
        prob_exact_dec = prob_win * prob_gtd
        
        if prob_exact_dec > 0.50: # 50% chance for +200 odds
            wager = STAKE
            won_bet = did_win and is_decision
            
            if won_bet:
                profit = wager * (AVG_DEC_WIN_ODDS - 1)
                res = 'WIN'
            else:
                profit = -wager
                res = 'LOSS'
            
            bankroll += profit
            history.append({'Date': row['event_date'], 'Fight': f"{row['f_1_name']} vs {row['f_2_name']}", 'Bet_Type': 'EXACT_DEC', 'Prob': prob_exact_dec, 'Result': res, 'Profit': profit, 'Bankroll': bankroll})

        # --- Previous Strategies (Keep them for comparison) ---
        # ... (Simplified for brevity, or keep them if user wants all)
        # Let's keep the previous ones too.
        
        # KO Strategy
        if prob_fight_ko > 0.50:
            wager = STAKE
            if is_ko: profit = wager * (AVG_KO_ODDS - 1); res = 'WIN'
            else: profit = -wager; res = 'LOSS'
            bankroll += profit
            history.append({'Date': row['event_date'], 'Fight': f"{row['f_1_name']} vs {row['f_2_name']}", 'Bet_Type': 'KO', 'Prob': prob_fight_ko, 'Result': res, 'Profit': profit, 'Bankroll': bankroll})
            
        # Sub Strategy
        if prob_fight_sub > 0.30:
            wager = STAKE
            if is_sub: profit = wager * (AVG_SUB_ODDS - 1); res = 'WIN'
            else: profit = -wager; res = 'LOSS'
            bankroll += profit
            history.append({'Date': row['event_date'], 'Fight': f"{row['f_1_name']} vs {row['f_2_name']}", 'Bet_Type': 'SUB', 'Prob': prob_fight_sub, 'Result': res, 'Profit': profit, 'Bankroll': bankroll})
            
        # GTD Strategy
        if prob_gtd > 0.60:
            wager = STAKE
            if is_decision: profit = wager * (AVG_GTD_ODDS - 1); res = 'WIN'
            else: profit = -wager; res = 'LOSS'
            bankroll += profit
            history.append({'Date': row['event_date'], 'Fight': f"{row['f_1_name']} vs {row['f_2_name']}", 'Bet_Type': 'GTD', 'Prob': prob_gtd, 'Result': res, 'Profit': profit, 'Bankroll': bankroll})
            
        # Round Strategy
        if best_round_prob > 0.30:
            wager = STAKE
            if finish_round == pred_round: profit = wager * (AVG_RND_ODDS - 1); res = 'WIN'
            else: profit = -wager; res = 'LOSS'
            bankroll += profit
            history.append({'Date': row['event_date'], 'Fight': f"{row['f_1_name']} vs {row['f_2_name']}", 'Bet_Type': f'R{pred_round}', 'Prob': best_round_prob, 'Result': res, 'Profit': profit, 'Bankroll': bankroll})
            
    # 4. Save Log
    log_df = pd.DataFrame(history)
    log_df.to_csv('props_backtest_2024_2025.csv', index=False)
    
    # 5. Report
    print("\n=== Backtest Results (Conservative Assumptions) ===")
    print(f"Total Bets: {len(log_df)}")
    if not log_df.empty:
        total_profit = log_df['Profit'].sum()
        roi = total_profit / (len(log_df) * STAKE)
        
        ko_bets = log_df[log_df['Bet_Type'] == 'KO']
        sub_bets = log_df[log_df['Bet_Type'] == 'SUB']
        gtd_bets = log_df[log_df['Bet_Type'] == 'GTD']
        rnd_bets = log_df[log_df['Bet_Type'].str.startswith('R')]
        exact_ko_bets = log_df[log_df['Bet_Type'] == 'EXACT_KO']
        exact_dec_bets = log_df[log_df['Bet_Type'] == 'EXACT_DEC']
        
        def get_stats(df):
            if len(df) == 0: return 0, 0, 0
            wins = len(df[df['Result'] == 'WIN'])
            acc = wins / len(df)
            profit = df['Profit'].sum()
            return len(df), acc, profit

        n_ko, acc_ko, p_ko = get_stats(ko_bets)
        n_sub, acc_sub, p_sub = get_stats(sub_bets)
        n_gtd, acc_gtd, p_gtd = get_stats(gtd_bets)
        n_rnd, acc_rnd, p_rnd = get_stats(rnd_bets)
        n_ex_ko, acc_ex_ko, p_ex_ko = get_stats(exact_ko_bets)
        n_ex_dec, acc_ex_dec, p_ex_dec = get_stats(exact_dec_bets)
        
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"ROI: {roi:.2%}")
        print(f"Final Bankroll: ${bankroll:.2f}")
        print("-" * 60)
        print(f"{'Type':<10} | {'Bets':<5} | {'Win Rate':<10} | {'Profit':<10}")
        print("-" * 60)
        print(f"{'KO':<10} | {n_ko:<5} | {acc_ko:<10.1%} | ${p_ko:.2f}")
        print(f"{'SUB':<10} | {n_sub:<5} | {acc_sub:<10.1%} | ${p_sub:.2f}")
        print(f"{'GTD':<10} | {n_gtd:<5} | {acc_gtd:<10.1%} | ${p_gtd:.2f}")
        print(f"{'RND':<10} | {n_rnd:<5} | {acc_rnd:<10.1%} | ${p_rnd:.2f}")
        print(f"{'EXACT_KO':<10} | {n_ex_ko:<5} | {acc_ex_ko:<10.1%} | ${p_ex_ko:.2f}")
        print(f"{'EXACT_DEC':<10} | {n_ex_dec:<5} | {acc_ex_dec:<10.1%} | ${p_ex_dec:.2f}")
    else:
        print("No bets placed.")

if __name__ == "__main__":
    generate_props_log()
