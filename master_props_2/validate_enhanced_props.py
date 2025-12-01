import pandas as pd
import numpy as np
import joblib
import json
import torch
import os
import sys

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'master_3')))
from master_3.api_utils import Master3Predictor
from master_3.models.opponent_adjustment import apply_opponent_adjustment

def validate_enhanced_trifecta():
    print("=== Master Props 2: Validating Enhanced Trifecta (2024-2025) ===")
    
    # 1. Load Data
    df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    test_mask = df['event_date'] >= '2024-01-01'
    test_df = df[test_mask].copy().reset_index(drop=True)
    
    print(f"Test Set: {len(test_df)} fights")
    
    # 2. Load Models
    print("Loading models...")
    # Master 3 Predictor (for Winner)
    m3 = Master3Predictor(base_dir='../master_3')
    
    # Enhanced Props
    model_finish = joblib.load('models/prop_enhanced_finish.pkl')
    model_method = joblib.load('models/prop_enhanced_method.pkl')
    model_round = joblib.load('models/prop_enhanced_round.pkl')
    
    # Features
    with open('../master_3/features_enhanced.json', 'r') as f:
        features = json.load(f)
        
    # Apply Opponent Adjustment to Test DF
    # We need to pass the full DF to apply_opponent_adjustment
    # It expects certain columns.
    # Let's check if test_df has them.
    # It should.
    stat_cols = [
        'slpm_15_f_1', 'sapm_15_f_1', 'td_avg_15_f_1', 'sub_avg_15_f_1',
        'slpm_15_f_2', 'sapm_15_f_2', 'td_avg_15_f_2', 'sub_avg_15_f_2'
    ]
    # Check if they exist
    missing_stats = [c for c in stat_cols if c not in test_df.columns]
    if missing_stats:
        print(f"Warning: Missing stat cols for adjustment: {missing_stats}")
        # Try to find them without _15?
        # master_3 uses _15 suffix usually.
        # If missing, maybe skip adjustment or fill 0.
        pass
        
    test_df_adj = apply_opponent_adjustment(test_df, stat_cols, elo_col='dynamic_elo')
    
    X_test = test_df_adj[features].fillna(0)
    
    # 3. Predict
    # Batch sequence prep
    f1_names = test_df['f_1_name'].values
    f2_names = test_df['f_2_name'].values
    
    seq1_list = []
    seq2_list = []
    
    for i in range(len(test_df)):
        seq1_list.append(m3.get_sequence(f1_names[i]))
        seq2_list.append(m3.get_sequence(f2_names[i]))
        
    seq1_arr = np.array(seq1_list)
    seq2_arr = np.array(seq2_list)
    
    # Scale Sequences
    # Reshape (N*Seq, D) -> Transform -> Reshape (N, Seq, D)
    N, S, D = seq1_arr.shape
    s1_flat = seq1_arr.reshape(-1, D)
    s2_flat = seq2_arr.reshape(-1, D)
    
    s1_scaled = m3.scaler.transform(s1_flat).reshape(N, S, D)
    s2_scaled = m3.scaler.transform(s2_flat).reshape(N, S, D)
    
    # Static Features for Siamese
    # We need to extract pairs
    f1_static = []
    f2_static = []
    for c1, c2 in m3.siamese_pairs:
        # Siamese pairs might be in X_test_props OR test_df_adj?
        # Siamese cols are usually raw stats.
        # X_test_props has features_enhanced.
        # Let's use test_df_adj to be safe, as it has everything.
        f1_static.append(test_df_adj[c1].fillna(0).values)
        f2_static.append(test_df_adj[c2].fillna(0).values)
        
    f1_static = np.array(f1_static).T # (N, D)
    f2_static = np.array(f2_static).T
    
    f1_static = m3.scaler.transform(f1_static)
    f2_static = m3.scaler.transform(f2_static)
    
    # Run Siamese
    sia_probs = []
    batch_size = 64
    m3.siamese_model.eval()
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            t_f1 = torch.FloatTensor(f1_static[i:end]).to(m3.device)
            t_f2 = torch.FloatTensor(f2_static[i:end]).to(m3.device)
            t_s1 = torch.FloatTensor(s1_scaled[i:end]).to(m3.device)
            t_s2 = torch.FloatTensor(s2_scaled[i:end]).to(m3.device)
            
            out = m3.siamese_model(t_f1, t_f2, t_s1, t_s2).cpu().numpy()
            sia_probs.extend(out)
            
    sia_probs = np.array(sia_probs).flatten()
    
    # Ensemble
    w = m3.params.get('ensemble_xgb_weight', 0.5)
    p_win = w * xgb_probs + (1 - w) * sia_probs
    
    # Props
    p_finish = model_finish.predict_proba(X_test_props)[:, 1]
    p_method = model_method.predict_proba(X_test_props) # [KO, Sub]
    p_round = model_round.predict_proba(X_test_props)   # [1,2,3,4,5]
    
    # 4. Calculate Accuracy
    correct_trifecta = 0
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # Winner
        if p_win[i] > 0.5:
            pred_winner = row['f_1_name']
            conf_w = p_win[i]
        else:
            pred_winner = row['f_2_name']
            conf_w = 1 - p_win[i]
            
        # Method
        prob_ko = p_finish[i] * p_method[i][0]
        prob_sub = p_finish[i] * p_method[i][1]
        prob_dec = 1 - p_finish[i]
        
        methods = {'KO/TKO': prob_ko, 'Submission': prob_sub, 'Decision': prob_dec}
        pred_method = max(methods, key=methods.get)
        
        # Round
        best_rnd_idx = np.argmax(p_round[i])
        pred_round = best_rnd_idx + 1
        
        # Actuals
        actual_winner = str(row.get('winner', 'Unknown'))
        actual_res = str(row['result']).lower()
        try:
            actual_round = int(row['finish_round'])
        except:
            actual_round = -1
            
        # Check
        is_win_correct = (pred_winner == actual_winner)
        
        is_method_correct = False
        if 'ko' in actual_res and pred_method == 'KO/TKO': is_method_correct = True
        if 'submission' in actual_res and pred_method == 'Submission': is_method_correct = True
        if 'decision' in actual_res and pred_method == 'Decision': is_method_correct = True
        
        is_round_correct = (pred_round == actual_round)
        
        # Trifecta Logic
        if pred_method == 'Decision':
            # If predicted decision, round doesn't matter (or is implicitly correct if it went distance)
            # But wait, if actual was Decision, round is usually 3 or 5.
            # If we predict Decision, we are predicting it goes the distance.
            # So is_round_correct is irrelevant?
            # In Phase 8, we treated Decision as a "Trifecta" if Winner + Method were correct.
            # Let's stick to that definition.
            is_trifecta = is_win_correct and is_method_correct
        else:
            # If predicted Finish, Round MUST be correct.
            is_trifecta = is_win_correct and is_method_correct and is_round_correct
            
        if is_trifecta:
            correct_trifecta += 1
            
    acc = correct_trifecta / len(test_df)
    print(f"\nEnhanced Trifecta Accuracy: {acc:.2%}")
    print(f"Baseline (Phase 8):       31.62%")
    print(f"Improvement:              {acc - 0.3162:+.2%}")

if __name__ == "__main__":
    validate_enhanced_trifecta()
