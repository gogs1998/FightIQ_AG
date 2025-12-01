import pandas as pd
import numpy as np

def calculate_chin_features(df):
    """
    Calculate Chin Health Decay scores.
    Formula: Score = 1.0 * (0.9 ^ ko_losses) * (0.98 ^ kd_absorbed)
    """
    df = df.copy()
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
    fighter_damage = {} # {fighter_name: {'ko_losses': 0, 'kd_absorbed': 0}}
    
    chin_scores_f1 = []
    chin_scores_f2 = []
    
    for idx, row in df.iterrows():
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        
        if f1 not in fighter_damage: fighter_damage[f1] = {'ko_losses': 0, 'kd_absorbed': 0}
        if f2 not in fighter_damage: fighter_damage[f2] = {'ko_losses': 0, 'kd_absorbed': 0}
        
        d1 = fighter_damage[f1]
        d2 = fighter_damage[f2]
        
        score1 = 1.0 * (0.9 ** d1['ko_losses']) * (0.98 ** d1['kd_absorbed'])
        score2 = 1.0 * (0.9 ** d2['ko_losses']) * (0.98 ** d2['kd_absorbed'])
        
        chin_scores_f1.append(score1)
        chin_scores_f2.append(score2)
        
        # Update stats AFTER the fight
        res = str(row.get('result', '')).lower()
        if not res: res = str(row.get('method', '')).lower()
        
        if 'ko' in res or 'tko' in res:
            if row['winner'] == f2: # F1 lost by KO
                fighter_damage[f1]['ko_losses'] += 1
            elif row['winner'] == f1: # F2 lost by KO
                fighter_damage[f2]['ko_losses'] += 1
                
        kd1 = row.get('f_1_kd', 0)
        kd2 = row.get('f_2_kd', 0)
        
        if pd.notna(kd1): fighter_damage[f2]['kd_absorbed'] += kd1
        if pd.notna(kd2): fighter_damage[f1]['kd_absorbed'] += kd2
        
    df['f_1_chin_score'] = chin_scores_f1
    df['f_2_chin_score'] = chin_scores_f2
    df['diff_chin_score'] = df['f_1_chin_score'] - df['f_2_chin_score']
    
    return df
