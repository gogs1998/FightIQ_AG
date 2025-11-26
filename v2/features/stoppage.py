import pandas as pd
import numpy as np

def calculate_stoppage_features(df):
    """
    Calculate features related to early stoppages.
    - finish_rate: % of wins that are KO/Sub
    - been_finished_rate: % of losses that are KO/Sub
    - avg_fight_time: Average duration of fights
    """
    df = df.copy()
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
    def is_finish(row):
        res_str = ""
        if 'result' in row and pd.notna(row['result']): res_str += str(row['result'])
        if 'method' in row and pd.notna(row['method']): res_str += str(row['method'])
        if 'result_details' in row and pd.notna(row['result_details']): res_str += str(row['result_details'])
        
        res_lower = res_str.lower()
        if 'ko' in res_lower or 'tko' in res_lower or 'submission' in res_lower:
            return True
        return False

    df['is_finish'] = df.apply(is_finish, axis=1)
    
    if 'fight_duration_minutes' not in df.columns:
        df['duration'] = np.nan
    else:
        df['duration'] = df['fight_duration_minutes']
        
    history = {} 
    
    f1_finish_rate = []
    f1_been_finished_rate = []
    f1_avg_time = []
    
    f2_finish_rate = []
    f2_been_finished_rate = []
    f2_avg_time = []
    
    for idx, row in df.iterrows():
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        winner = row['winner']
        is_fin = row['is_finish']
        dur = row.get('duration', 0)
        if pd.isna(dur): dur = 0
        
        h1 = history.get(f1, {'wins': 0, 'wins_finish': 0, 'losses': 0, 'losses_finish': 0, 'total_time': 0, 'n_fights': 0})
        h2 = history.get(f2, {'wins': 0, 'wins_finish': 0, 'losses': 0, 'losses_finish': 0, 'total_time': 0, 'n_fights': 0})
        
        # F1 Stats
        f1_finish_rate.append(h1['wins_finish'] / h1['wins'] if h1['wins'] > 0 else 0.0)
        f1_been_finished_rate.append(h1['losses_finish'] / h1['losses'] if h1['losses'] > 0 else 0.0)
        f1_avg_time.append(h1['total_time'] / h1['n_fights'] if h1['n_fights'] > 0 else 0.0)
            
        # F2 Stats
        f2_finish_rate.append(h2['wins_finish'] / h2['wins'] if h2['wins'] > 0 else 0.0)
        f2_been_finished_rate.append(h2['losses_finish'] / h2['losses'] if h2['losses'] > 0 else 0.0)
        f2_avg_time.append(h2['total_time'] / h2['n_fights'] if h2['n_fights'] > 0 else 0.0)
            
        # Update History
        h1['n_fights'] += 1
        h1['total_time'] += dur
        if winner == f1:
            h1['wins'] += 1
            if is_fin: h1['wins_finish'] += 1
        elif winner == f2:
            h1['losses'] += 1
            if is_fin: h1['losses_finish'] += 1
        
        h2['n_fights'] += 1
        h2['total_time'] += dur
        if winner == f2:
            h2['wins'] += 1
            if is_fin: h2['wins_finish'] += 1
        elif winner == f1:
            h2['losses'] += 1
            if is_fin: h2['losses_finish'] += 1
            
        history[f1] = h1
        history[f2] = h2
        
    df['f_1_finish_rate'] = f1_finish_rate
    df['f_1_been_finished_rate'] = f1_been_finished_rate
    df['f_1_avg_time'] = f1_avg_time
    
    df['f_2_finish_rate'] = f2_finish_rate
    df['f_2_been_finished_rate'] = f2_been_finished_rate
    df['f_2_avg_time'] = f2_avg_time
    
    df['diff_finish_rate'] = df['f_1_finish_rate'] - df['f_2_finish_rate']
    df['diff_been_finished_rate'] = df['f_1_been_finished_rate'] - df['f_2_been_finished_rate']
    df['diff_avg_time'] = df['f_1_avg_time'] - df['f_2_avg_time']
    
    return df
