import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

def generate_full_predictions():
    print("=== Master Props: Generating Full Prediction Log (2024-2025) ===")
    
    # 1. Load Data
    try:
        df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    except:
        df = pd.read_csv('data/training_data_enhanced.csv')
        
    # Manual Opponent Adjustment
    print("Applying manual opponent adjustment...")
    elo_col = 'dynamic_elo'
    avg_elo = df[f'{elo_col}_f1'].mean()
    
    stat_cols = [
        'slpm_15_f_1', 'sapm_15_f_1', 'td_avg_15_f_1', 'sub_avg_15_f_1',
        'slpm_15_f_2', 'sapm_15_f_2', 'td_avg_15_f_2', 'sub_avg_15_f_2'
    ]
    
    df_adj = df.copy()
    for col in stat_cols:
        if col.startswith('f_1_') or col.endswith('_f_1'):
            opp_elo = df[f'{elo_col}_f2']
            df_adj[f'{col}_adj'] = df[col] * (opp_elo / avg_elo)
        elif col.startswith('f_2_') or col.endswith('_f_2'):
            opp_elo = df[f'{elo_col}_f1']
            df_adj[f'{col}_adj'] = df[col] * (opp_elo / avg_elo)
            
    print("Adjusted columns created:", [c for c in df_adj.columns if '_adj' in c])
    
    # Filter for 2024-2025
    df_adj['event_date'] = pd.to_datetime(df_adj['event_date'])
    mask_test = df_adj['event_date'] >= '2024-01-01'
    test_df = df_adj[mask_test].copy().reset_index(drop=True)
    
    # 2. Load Features
    with open('../master_3/features_selected.json', 'r') as f:
        features_main = json.load(f)
        
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
            
    # Add prop features to features_main because the model apparently expects them too!
    prop_features = ['f_1_chin_score', 'f_2_chin_score', 'diff_chin_score']
    for f in prop_features:
        if f not in features_main:
            features_main.append(f)
            
    features_props = features_main.copy() # Now they are identical, which is fine
            
    features_props = features_main.copy()
    prop_features = ['f_1_chin_score', 'f_2_chin_score', 'diff_chin_score']
    for f in prop_features:
        if f not in features_props:
            features_props.append(f)
            
    # Clean column names
    df_adj.columns = df_adj.columns.str.strip()
    features_props = [f.strip() for f in features_props]
    
    # 3. Train Fresh Models (Winner, Finish, Method, Round)
    print("Training fresh models on 2010-2023 data...")
    train_mask = df_adj['event_date'] < '2024-01-01'
    train_df = df_adj[train_mask].copy()
    
    # --- Targets ---
    def get_winner_target(row):
        if row['winner'] == row['f_1_name']: return 1
        if row['winner'] == row['f_2_name']: return 0
        return -1 
        
    def get_finish_target(row):
        res = str(row['result']).lower()
        if 'decision' in res: return 0
        if 'draw' in res or 'no contest' in res: return -1
        return 1
        
    def get_method_target(row):
        res = str(row['result']).lower()
        if 'ko' in res or 'tko' in res: return 0 # KO
        if 'submission' in res: return 1 # Sub
        return -1 # Ignore Decision/Other for this model
        
    def get_round_target(row):
        try:
            r = int(row['finish_round'])
            if 1 <= r <= 5: return r - 1 # 0-indexed class
            return -1
        except:
            return -1

    train_df['win_target'] = train_df.apply(get_winner_target, axis=1)
    train_df['finish_target'] = train_df.apply(get_finish_target, axis=1)
    train_df['method_target'] = train_df.apply(get_method_target, axis=1)
    train_df['round_target'] = train_df.apply(get_round_target, axis=1)
    
    # Common X_train
    X_train_full = train_df[[c for c in features_props if c in train_df.columns]].fillna(0)
    
    import xgboost as xgb
    
    # Winner Model (Boosted parameters to try and match 74% baseline)
    mask_win = train_df['win_target'] != -1
    model_win = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model_win.fit(X_train_full[mask_win], train_df.loc[mask_win, 'win_target'])
    print("Winner Model Trained (Boosted).")
    
    # Finish Model
    mask_fin = train_df['finish_target'] != -1
    model_finish = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    model_finish.fit(X_train_full[mask_fin], train_df.loc[mask_fin, 'finish_target'])
    print("Finish Model Trained.")
    
    # Method Model (Only train on finishes)
    mask_meth = train_df['method_target'] != -1
    model_method = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    model_method.fit(X_train_full[mask_meth], train_df.loc[mask_meth, 'method_target'])
    print("Method Model Trained.")
    
    # Round Model (Only train on finishes)
    mask_rnd = train_df['round_target'] != -1
    model_round = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, objective='multi:softprob', num_class=5)
    model_round.fit(X_train_full[mask_rnd], train_df.loc[mask_rnd, 'round_target'])
    print("Round Model Trained.")

    # 4. Generate Predictions
    print("Predicting...")
    # Create X_props using EXACTLY the same columns as X_train
    X_props = pd.DataFrame(index=test_df.index)
    for col in X_train_full.columns:
        if col in test_df.columns:
            X_props[col] = test_df[col]
        else:
            X_props[col] = 0.0
            
    p_win = model_win.predict_proba(X_props)[:, 1]
    p_finish = model_finish.predict_proba(X_props)[:, 1]
    p_method_probs = model_method.predict_proba(X_props) # [N, 2] (KO, Sub)
    p_round_probs = model_round.predict_proba(X_props)   # [N, 5]
    
    log_data = []
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # --- Winner ---
        prob_w = p_win[i]
        if prob_w > 0.5:
            pred_winner = row['f_1_name']
            prob_winner = prob_w
        else:
            pred_winner = row['f_2_name']
            prob_winner = 1 - prob_w
            
        # --- Method ---
        # P(Method) = P(Finish) * P(Method|Finish) vs P(Decision)
        prob_ko = p_finish[i] * p_method_probs[i][0]
        prob_sub = p_finish[i] * p_method_probs[i][1]
        prob_dec = 1 - p_finish[i]
        
        # Pick max method
        methods = {'KO/TKO': prob_ko, 'Submission': prob_sub, 'Decision': prob_dec}
        pred_method = max(methods, key=methods.get)
        prob_method = methods[pred_method]
        
        # --- Round ---
        best_round_idx = np.argmax(p_round_probs[i])
        pred_round = best_round_idx + 1
        prob_round = p_round_probs[i][best_round_idx]
        
        # --- Actual Result ---
        actual_res = str(row['result'])
        finish_round = row['finish_round']
        if pd.isna(finish_round): finish_round = -1
        else: finish_round = int(finish_round)
        
        # Check Correctness
        actual_winner = str(row.get('winner', 'Unknown'))
        correct_winner = (pred_winner == actual_winner)
        
        # Method
        correct_method = False
        if 'ko' in actual_res.lower() and pred_method == 'KO/TKO': correct_method = True
        if 'submission' in actual_res.lower() and pred_method == 'Submission': correct_method = True
        if 'decision' in actual_res.lower() and pred_method == 'Decision': correct_method = True
        
        # Round
        correct_round = (pred_round == finish_round)
        
        # Trifecta (Perfect Prediction)
        # Note: If Method is Decision, Round MUST be 3 or 5 (depending on fight).
        # But our Round model predicts 1-5.
        # If Pred Method is Decision, we should probably ignore Round or assume it matches "Distance".
        # For now, let's be strict: Winner + Method + Round must all match.
        # Exception: If Method is Decision, Round is usually irrelevant/implicit, but let's stick to the data.
        correct_trifecta = correct_winner and correct_method and correct_round
        
        log_data.append({
            'Date': row['event_date'].strftime('%Y-%m-%d'),
            'Fight': f"{row['f_1_name']} vs {row['f_2_name']}",
            'Pred_Winner': pred_winner,
            'Prob_Win': f"{prob_winner:.1%}",
            'Pred_Method': pred_method,
            'Prob_Method': f"{prob_method:.1%}",
            'Pred_Round': pred_round,
            'Prob_Round': f"{prob_round:.1%}",
            'Actual_Winner': actual_winner,
            'Actual_Result': actual_res,
            'Actual_Round': finish_round,
            'Correct_Win': correct_winner,
            'Correct_Method': correct_method,
            'Correct_Round': correct_round,
            'Correct_Trifecta': correct_trifecta
        })
        
    # Save
    full_df = pd.DataFrame(log_data)
    full_df.to_csv('full_predictions_2024_2025.csv', index=False)
    print(f"Saved {len(full_df)} predictions to full_predictions_2024_2025.csv")
    
    # Save Perfect Predictions
    perfect_df = full_df[full_df['Correct_Trifecta'] == True]
    perfect_df.to_csv('perfect_predictions_2024_2025.csv', index=False)
    print(f"Saved {len(perfect_df)} PERFECT predictions to perfect_predictions_2024_2025.csv")
    
    # Print Summary Accuracy
    print("\n=== Accuracy Summary ===")
    print(f"Winner Accuracy:   {full_df['Correct_Win'].mean():.2%}")
    print(f"Method Accuracy:   {full_df['Correct_Method'].mean():.2%}")
    print(f"Round Accuracy:    {full_df['Correct_Round'].mean():.2%}")
    print(f"Trifecta Accuracy: {full_df['Correct_Trifecta'].mean():.2%} (Winner + Method + Round)")
    
    # --- Trifecta ROI Simulation ---
    print("\n=== Trifecta Betting Simulation (Theoretical) ===")
    # Assumptions
    ODDS_DEC = 3.00  # +200
    ODDS_FINISH = 19.00 # +1800 (Conservative average for specific round KO/Sub)
    STAKE = 100
    
    bankroll = 0
    total_wagered = 0
    wins = 0
    
    for i, row in full_df.iterrows():
        total_wagered += STAKE
        
        # Determine Odds
        if row['Pred_Method'] == 'Decision':
            odds = ODDS_DEC
        else:
            odds = ODDS_FINISH
            
        if row['Correct_Trifecta']:
            profit = STAKE * (odds - 1)
            bankroll += profit
            wins += 1
        else:
            bankroll -= STAKE
            
    roi = bankroll / total_wagered
    print(f"Total Bets: {len(full_df)}")
    print(f"Trifecta Wins: {wins}")
    print(f"Total Profit: ${bankroll:,.2f}")
    print(f"ROI: {roi:.2%}")
    print(f"Assumptions: Decision={ODDS_DEC} (+200), Exact Finish={ODDS_FINISH} (+1800)")

if __name__ == "__main__":
    generate_full_predictions()
