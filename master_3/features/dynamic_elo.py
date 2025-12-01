import pandas as pd
import numpy as np

class DynamicEloTracker:
    def __init__(self, base_k=20, placement_k_factor=2.0, finish_k_factor=1.5, placement_fights=5):
        self.ratings = {} # fighter_name -> rating
        self.fight_counts = {} # fighter_name -> n_fights
        self.base_k = base_k
        self.placement_k_factor = placement_k_factor
        self.finish_k_factor = finish_k_factor
        self.placement_fights = placement_fights

    def get_rating(self, fighter):
        return self.ratings.get(fighter, 1500.0)

    def get_k(self, fighter, is_finish=False):
        k = self.base_k
        
        # Placement Bonus
        n = self.fight_counts.get(fighter, 0)
        if n < self.placement_fights:
            k *= self.placement_k_factor
            
        # Finish Bonus
        if is_finish:
            k *= self.finish_k_factor
            
        return k

    def update(self, f1, f2, winner, result_type=None):
        r1 = self.get_rating(f1)
        r2 = self.get_rating(f2)
        
        # Expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Actual scores
        if winner == f1:
            s1, s2 = 1.0, 0.0
        elif winner == f2:
            s1, s2 = 0.0, 1.0
        else:
            s1, s2 = 0.5, 0.5 # Draw
            
        # Determine if finish
        is_finish = False
        if result_type:
            res_lower = str(result_type).lower()
            if 'ko' in res_lower or 'tko' in res_lower or 'submission' in res_lower:
                is_finish = True
        
        # Calculate K
        k1 = self.get_k(f1, is_finish)
        k2 = self.get_k(f2, is_finish)
        
        # Update
        new_r1 = r1 + k1 * (s1 - e1)
        new_r2 = r2 + k2 * (s2 - e2)
        
        self.ratings[f1] = new_r1
        self.ratings[f2] = new_r2
        
        self.fight_counts[f1] = self.fight_counts.get(f1, 0) + 1
        self.fight_counts[f2] = self.fight_counts.get(f2, 0) + 1
        
        return r1, r2 # Return PRE-FIGHT ratings

def calculate_dynamic_elo(df):
    """
    Calculate dynamic Elo for the entire dataframe.
    Assumes df is sorted by date.
    """
    tracker = DynamicEloTracker()
    
    elo_f1_series = []
    elo_f2_series = []
    
    # Ensure sorted
    if 'event_date' in df.columns:
        df = df.sort_values('event_date')
        
    for idx, row in df.iterrows():
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        winner = row['winner']
        
        # Try to find result type
        res_type = row.get('result') or row.get('method') or row.get('result_details')
        
        # Get current ratings (pre-fight)
        r1, r2 = tracker.update(f1, f2, winner, res_type)
        
        elo_f1_series.append(r1)
        elo_f2_series.append(r2)
        
    df_out = df.copy()
    df_out['dynamic_elo_f1'] = elo_f1_series
    df_out['dynamic_elo_f2'] = elo_f2_series
    df_out['diff_dynamic_elo'] = df_out['dynamic_elo_f1'] - df_out['dynamic_elo_f2']
    
    return df_out
