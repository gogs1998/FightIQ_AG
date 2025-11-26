import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import random

def generate_fake_matches(df):
    """
    Generate fake matchups by shuffling opponents within the same weight class and era.
    """
    df = df.copy()
    if 'weight_class' not in df.columns:
        # Fallback if weight_class missing: use simple random shuffle within small time blocks
        print("Warning: weight_class not found, using simple shuffle.")
        df['weight_class'] = 'Unknown'

    # Ensure datetime
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    fake_matches = []
    
    # Group by time block (e.g., 6 months) and weight class to get realistic potential matchups
    # Using 'year' for broader pool if events are sparse, or just shuffle within event if enough fights
    # Let's try shuffling within +/- 90 days
    
    df = df.sort_values('event_date')
    
    # Simple approach: For each fight, pick a random opponent from the same weight class 
    # active within +/- 180 days.
    
    # Get list of (fighter, date, weight_class)
    # Actually, simpler: Just shuffle 'f_2_name' (and associated stats) within groups.
    
    # We need to pair f1 features with f2 features.
    # Let's extract f1 columns and f2 columns.
    
    cols = df.columns
    f1_cols = [c for c in cols if c.startswith('f_1_') or c.endswith('_f_1')]
    f2_cols = [c for c in cols if c.startswith('f_2_') or c.endswith('_f_2')]
    meta_cols = ['event_date', 'weight_class', 'fight_id']
    
    # We need to be careful about 'diff_' columns. We can't use them directly for fakes 
    # because they are pre-calculated. We would need to recalculate them.
    # For propensity model, let's just use raw f1/f2 features to predict.
    
    # Actually, to train the propensity model, we need features that represent the "matchup quality".
    # e.g. Rank diff, Elo diff, Age diff.
    # So we DO need to recalculate diffs for the fake pairs.
    
    # Strategy:
    # 1. Create a pool of "fighter instances" (fighter features at a specific date).
    # 2. Randomly pair them.
    # 3. Calculate "matchup features" (diffs).
    
    # Let's simplify. We will only use a few key features for propensity:
    # Elo, Age, Reach (if avail), Win Streak.
    
    # Helper to get value
    def get_val(row, col_base, f_num):
        # Try f_{n}_{col} or {col}_f_{n}
        c1 = f"f_{f_num}_{col_base}"
        c2 = f"{col_base}_f_{f_num}"
        if c1 in row: return row[c1]
        if c2 in row: return row[c2]
        return np.nan

    fakes = []
    
    # Iterate through time blocks
    start_date = df['event_date'].min()
    end_date = df['event_date'].max()
    
    # Create 6-month blocks
    dates = pd.date_range(start=start_date, end=end_date, freq='180D')
    
    for i in range(len(dates)-1):
        d_start = dates[i]
        d_end = dates[i+1]
        
        mask = (df['event_date'] >= d_start) & (df['event_date'] < d_end)
        block = df[mask]
        
        for wc, g in block.groupby('weight_class'):
            if len(g) < 2: continue
            
            # Create list of fighter dicts
            fighters = []
            for idx, row in g.iterrows():
                # Fighter 1
                f1 = {
                    'name': row['f_1_name'],
                    'elo': get_val(row, 'elo', 1),
                    'age': get_val(row, 'age', 1),
                    'streak': get_val(row, 'streak', 1) if 'streak_3_f_1' not in row else row.get('streak_3_f_1', 0), # approximate
                    'rank': row.get('f_1_rank', 99) # Assuming rank might not be there, handle later
                }
                fighters.append(f1)
                
                # Fighter 2
                f2 = {
                    'name': row['f_2_name'],
                    'elo': get_val(row, 'elo', 2),
                    'age': get_val(row, 'age', 2),
                    'streak': get_val(row, 'streak', 2) if 'streak_3_f_2' not in row else row.get('streak_3_f_2', 0),
                    'rank': row.get('f_2_rank', 99)
                }
                fighters.append(f2)
            
            # Shuffle and pair
            random.shuffle(fighters)
            # Create pairs (0,1), (2,3)...
            for k in range(0, len(fighters)-1, 2):
                fa = fighters[k]
                fb = fighters[k+1]
                
                # Skip if same person (unlikely but possible if data dirty)
                if fa['name'] == fb['name']: continue
                
                # Calculate diffs
                fakes.append({
                    'diff_elo': abs(fa['elo'] - fb['elo']),
                    'diff_age': abs(fa['age'] - fb['age']),
                    'is_real': 0
                })

    # Process Real matches
    reals = []
    for idx, row in df.iterrows():
        e1 = get_val(row, 'elo', 1)
        e2 = get_val(row, 'elo', 2)
        a1 = get_val(row, 'age', 1)
        a2 = get_val(row, 'age', 2)
        
        reals.append({
            'diff_elo': abs(e1 - e2),
            'diff_age': abs(a1 - a2),
            'is_real': 1
        })
        
    df_fake = pd.DataFrame(fakes)
    df_real = pd.DataFrame(reals)
    
    return pd.concat([df_real, df_fake], ignore_index=True)

def fit_propensity_model(df_all):
    """
    Train logistic regression to predict is_real.
    """
    df_all = df_all.dropna()
    X = df_all[['diff_elo', 'diff_age']]
    y = df_all['is_real']
    
    model = LogisticRegression(class_weight='balanced')
    model.fit(X, y)
    
    return model

def calculate_weights(df, model):
    """
    Calculate sample weights for real data.
    Weight = 1 / P(is_real)
    """
    # Extract features for prediction
    # We need to construct the same features as trained on
    
    def get_val(row, col_base, f_num):
        c1 = f"f_{f_num}_{col_base}"
        c2 = f"{col_base}_f_{f_num}"
        if c1 in row: return row[c1]
        if c2 in row: return row[c2]
        return np.nan

    data = []
    for idx, row in df.iterrows():
        e1 = get_val(row, 'elo', 1)
        e2 = get_val(row, 'elo', 2)
        a1 = get_val(row, 'age', 1)
        a2 = get_val(row, 'age', 2)
        data.append({
            'diff_elo': abs(e1 - e2),
            'diff_age': abs(a1 - a2)
        })
        
    X = pd.DataFrame(data)
    X = X.fillna(0) # Handle NaNs
    
    probs = model.predict_proba(X)[:, 1] # P(is_real)
    
    # Clip probabilities to avoid extreme weights
    probs = np.clip(probs, 0.05, 0.95)
    
    weights = 1.0 / probs
    
    return weights, probs
