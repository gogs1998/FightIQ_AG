import pandas as pd
import numpy as np

def calculate_common_opponent_features(df):
    """
    Calculate features based on common opponents (MMA Math).
    - triangle_score: Sum of comparative results against common opponents.
    - n_common_opponents: Count of common opponents.
    - common_win_pct_diff: Difference in win % against common opponents.
    """
    df = df.copy()
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
    history = {} # fighter_id -> {opponent_id: [results]}
    
    triangle_scores = []
    n_common_list = []
    win_pct_diff_list = []
    
    for idx, row in df.iterrows():
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        winner = row['winner']
        
        # 1. Calculate Features based on PAST history
        hist_f1 = history.get(f1, {})
        hist_f2 = history.get(f2, {})
        
        common_opps = set(hist_f1.keys()) & set(hist_f2.keys())
        n_common = len(common_opps)
        
        score = 0
        w1_common = 0
        w2_common = 0
        total_common_fights_f1 = 0
        total_common_fights_f2 = 0
        
        if n_common > 0:
            for opp in common_opps:
                res1 = hist_f1[opp]
                res2 = hist_f2[opp]
                
                avg1 = sum(res1) / len(res1)
                avg2 = sum(res2) / len(res2)
                
                if avg1 > avg2:
                    score += 1
                elif avg2 > avg1:
                    score -= 1
                    
                w1_common += sum(res1)
                total_common_fights_f1 += len(res1)
                
                w2_common += sum(res2)
                total_common_fights_f2 += len(res2)
                
        triangle_scores.append(score)
        n_common_list.append(n_common)
        
        if total_common_fights_f1 > 0 and total_common_fights_f2 > 0:
            pct1 = w1_common / total_common_fights_f1
            pct2 = w2_common / total_common_fights_f2
            win_pct_diff_list.append(pct1 - pct2)
        else:
            win_pct_diff_list.append(0.0)
            
        # 2. Update History with CURRENT fight
        if winner == f1:
            r1, r2 = 1.0, 0.0
        elif winner == f2:
            r1, r2 = 0.0, 1.0
        else:
            r1, r2 = 0.5, 0.5
            
        if f1 not in history: history[f1] = {}
        if f2 not in history[f1]: history[f1][f2] = []
        history[f1][f2].append(r1)
        
        if f2 not in history: history[f2] = {}
        if f1 not in history[f2]: history[f2][f1] = []
        history[f2][f1].append(r2)
        
    df['triangle_score'] = triangle_scores
    df['n_common_opponents'] = n_common_list
    df['common_win_pct_diff'] = win_pct_diff_list
    
    return df
