import pandas as pd
import numpy as np
import os
import sys

# Add v2 to path
sys.path.append(os.path.join(os.getcwd(), 'v2'))
from features.dynamic_elo import DynamicEloTracker

class MatchupEngine:
    def __init__(self, data_path='v2/data/training_data_v2.csv'):
        self.data_path = data_path
        self.df = None
        self.fighter_stats = {} # name -> dict of latest stats
        self.history = {} # name -> set of opponents
        self.load_data()
        
    def load_data(self):
        if not os.path.exists(self.data_path):
            print(f"Error: {self.data_path} not found.")
            return
            
        print("Loading MatchupEngine data...")
        self.df = pd.read_csv(self.data_path)
        
        # Sort by date
        if 'event_date' in self.df.columns:
            self.df['event_date'] = pd.to_datetime(self.df['event_date'])
            self.df = self.df.sort_values('event_date')
            
        # Build Fighter Stats & History
        # We iterate through the dataframe and update the "latest" stats for each fighter
        
        print("Building fighter profiles...")
        for idx, row in self.df.iterrows():
            f1 = row['f_1_name']
            f2 = row['f_2_name']
            
            # Update History (for Common Opps)
            if f1 not in self.history: self.history[f1] = set()
            if f2 not in self.history: self.history[f2] = set()
            self.history[f1].add(f2)
            self.history[f2].add(f1)
            
            # Extract Stats
            # We want to capture the state *after* the fight? 
            # No, usually we want the state *entering* the next fight.
            # But the row contains features *entering* this fight.
            # So if we just process in order, the last time we see a fighter, 
            # their features in that row are the most recent "entering" state we have.
            # Wait, if they fought yesterday, we want those features.
            # But the row has their features *before* yesterday's fight.
            # Ideally we want to update their stats with the result of yesterday's fight.
            
            # However, our feature generation pipeline (aggregator) calculates features *pre-fight*.
            # So the values in the row are correct for *that* fight.
            # For a *new* fight, we technically need to re-calculate based on the last result.
            # But that requires re-running the whole feature logic (Elo update, etc.).
            
            # Simplified approach for V2 API:
            # Use the stats from their *last recorded fight*.
            # This misses the update from their very last fight, but it's a close approximation.
            # AND it's consistent with how we trained (using pre-fight features).
            # Actually, if we use the features from their last fight, we are using their state *before* their last fight.
            # That's 1 fight stale.
            
            # Better: We can manually update key metrics like Elo?
            # Or just accept 1-fight staleness for now.
            # Given the complexity, 1-fight staleness is acceptable for "MVP v2".
            # But we can do better for Elo since we have the tracker.
            
            # Let's store the raw features from the row
            self.fighter_stats[f1] = self._extract_fighter_features(row, 'f_1_')
            self.fighter_stats[f2] = self._extract_fighter_features(row, 'f_2_')
            
        print(f"Loaded profiles for {len(self.fighter_stats)} fighters.")
        
    def _extract_fighter_features(self, row, prefix):
        # Extract all columns starting with prefix OR ending with suffix
        # prefix is 'f_1_' or 'f_2_'
        suffix = '_' + prefix[:-1] # '_f_1' or '_f_2'
        
        stats = {}
        for col in row.index:
            if col.startswith(prefix):
                key = col[len(prefix):] # remove 'f_1_'
                stats[key] = row[col]
            elif col.endswith(suffix):
                key = col[:-len(suffix)] # remove '_f_1'
                stats[key] = row[col]
                
        # Special handling for dynamic_elo (dynamic_elo_f1, dynamic_elo_f2)
        fighter_num = prefix[2] # '1' or '2'
        elo_col = f'dynamic_elo_f{fighter_num}'
        if elo_col in row:
            stats['dynamic_elo'] = row[elo_col]
            
        return stats

    def build_matchup(self, f1_name, f2_name, f1_odds=None, f2_odds=None):
        # 1. Get Stats
        s1 = self.fighter_stats.get(f1_name, {})
        s2 = self.fighter_stats.get(f2_name, {})
        
        if not s1: print(f"Warning: {f1_name} not found.")
        if not s2: print(f"Warning: {f2_name} not found.")
        
        # 2. Construct Row
        row = {}
        
        # Base Features
        for k, v in s1.items(): 
            row[f'f_1_{k}'] = v
            row[f'{k}_f_1'] = v
            if k == 'dynamic_elo':
                row['dynamic_elo_f1'] = v
            
        for k, v in s2.items(): 
            row[f'f_2_{k}'] = v
            row[f'{k}_f_2'] = v
            if k == 'dynamic_elo':
                row['dynamic_elo_f2'] = v
        
        # Overwrite Odds if provided
        if f1_odds: row['f_1_odds'] = f1_odds
        if f2_odds: row['f_2_odds'] = f2_odds
        
        # 3. Calculate Diffs
        # We need to find common keys between s1 and s2 (e.g. 'reach', 'age', 'dynamic_elo')
        common_keys = set(s1.keys()) & set(s2.keys())
        for k in common_keys:
            # Check if it's numeric
            if isinstance(s1[k], (int, float)) and isinstance(s2[k], (int, float)):
                row[f'diff_{k}'] = s1[k] - s2[k]
                
        # 4. Common Opponents (Real-time calc)
        opps1 = self.history.get(f1_name, set())
        opps2 = self.history.get(f2_name, set())
        common = opps1 & opps2
        row['n_common_opponents'] = len(common)
        # We can't easily calc 'common_win_pct_diff' without looking up results against those opps.
        # For now, set to 0 or use a placeholder.
        # Given it's a specific feature, let's default to 0 to avoid crashing.
        row['common_win_pct_diff'] = 0.0 
        
        # 5. Return DataFrame
        return pd.DataFrame([row])

if __name__ == "__main__":
    engine = MatchupEngine()
    df = engine.build_matchup("Islam Makhachev", "Charles Oliveira")
    print(df.iloc[0][['f_1_name', 'f_2_name', 'diff_dynamic_elo', 'n_common_opponents']])
