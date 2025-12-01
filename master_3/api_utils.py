import pandas as pd
import numpy as np
import joblib
import json
import torch
import os
import sys

# Ensure we can import from local models
# If running from master_3/api_utils.py, we need to handle paths carefully.
# But usually this is imported.
# Let's assume 'models' is a package available in path (via sys.path.append in api.py)

try:
    from models import SiameseMatchupNet, FightSequenceEncoder, prepare_siamese_data
    from models.sequence_model import prepare_sequences
except ImportError:
    # Fallback if models is not in root path but relative
    from .models import SiameseMatchupNet, FightSequenceEncoder, prepare_siamese_data
    from .models.sequence_model import prepare_sequences

class Master3Predictor:
    def __init__(self, base_dir='master_3'):
        self.base_dir = base_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_resources()
        self.build_history()
        
    def load_resources(self):
        print("Loading Master 3 resources...")
        
        # Features
        feat_path = os.path.join(self.base_dir, 'features_selected.json')
        if not os.path.exists(feat_path):
            print("features_selected.json not found, using enhanced.")
            feat_path = os.path.join(self.base_dir, 'features_enhanced.json')
            
        print(f"Loading features from: {feat_path}")
        with open(feat_path, 'r') as f:
            self.features = json.load(f)
        print(f"Loaded {len(self.features)} features.")
            
        # Params
        with open(os.path.join(self.base_dir, 'params.json'), 'r') as f:
            self.params = json.load(f)['best_params']
            
        # Models
        self.xgb_model = joblib.load(os.path.join(self.base_dir, 'models/xgb_master3.pkl'))
        self.finish_model = joblib.load(os.path.join(self.base_dir, 'models/finish_master3.pkl'))
        self.scaler = joblib.load(os.path.join(self.base_dir, 'models/siamese_scaler.pkl'))
        
        # Siamese
        # Check for siamese_cols.json
        sia_cols_path = os.path.join(self.base_dir, 'models/siamese_cols.json')
        if os.path.exists(sia_cols_path):
            print(f"Loading Siamese cols from {sia_cols_path}")
            with open(sia_cols_path, 'r') as f:
                self.siamese_pairs_cols = json.load(f) # These are f1 cols
                
            # Construct pairs
            self.siamese_pairs = []
            for c1 in self.siamese_pairs_cols:
                # Find partner
                c2 = None
                if c1.startswith('f_1_'): c2 = c1.replace('f_1_', 'f_2_')
                elif '_f_1' in c1: c2 = c1.replace('_f_1', '_f_2')
                
                if c2:
                    self.siamese_pairs.append((c1, c2))
                else:
                    print(f"Warning: Could not find partner for {c1}")
                    
            self.siamese_input_dim = len(self.siamese_pairs)
            print(f"Siamese Input Dim (from json): {self.siamese_input_dim}")
            
            # Hack for 12 vs 11 mismatch
            if self.siamese_input_dim == 12:
                # Try dropping odds if it's there
                odds_col = 'f_1_odds'
                if odds_col in self.siamese_pairs_cols:
                    print("Dropping odds from Siamese cols to match dim 11...")
                    self.siamese_pairs = [p for p in self.siamese_pairs if p[0] != odds_col]
                    self.siamese_input_dim = len(self.siamese_pairs)
                    
        else:
            # Infer dims using prepare_siamese_data on a dummy DF
            dummy_df = pd.DataFrame(columns=self.features)
            _, _, self.siamese_input_dim, self.siamese_pairs_cols = prepare_siamese_data(dummy_df, self.features)
            
            # Re-implementing pair logic to match prepare_siamese_data exactly:
            self.siamese_pairs = self._get_feature_pairs_robust(self.features)
            self.siamese_input_dim = len(self.siamese_pairs)
            print(f"Found {self.siamese_input_dim} pairs (inferred)")
        
        # Sequence Dim
        self.seq_input_dim = self.siamese_input_dim 
        
        print(f"Final Siamese Input Dim: {self.siamese_input_dim}")
        
        self.siamese_model = SiameseMatchupNet(
            self.siamese_input_dim, 
            seq_input_dim=self.seq_input_dim, 
            hidden_dim=128 # Force 128 to match checkpoint
        ).to(self.device)
        
        state_path = os.path.join(self.base_dir, 'models/siamese_master3.pth')
        if os.path.exists(state_path):
            self.siamese_model.load_state_dict(torch.load(state_path, map_location=self.device))
            self.siamese_model.eval()
        else:
            print("Warning: Siamese weights not found.")
            self.siamese_model = None

    def _get_feature_pairs_robust(self, feature_cols):
        # Matches prepare_siamese_data logic from models/__init__.py
        pairs = set()
        all_cols = set(feature_cols)
        
        for feat in feature_cols:
            base = None
            f1_col = None
            f2_col = None
            
            if feat.startswith('diff_'):
                base = feat[5:] 
                if f"f_1_{base}" in all_cols and f"f_2_{base}" in all_cols:
                    f1_col = f"f_1_{base}"
                    f2_col = f"f_2_{base}"
                elif f"{base}_f_1" in all_cols and f"{base}_f_2" in all_cols:
                    f1_col = f"{base}_f_1"
                    f2_col = f"{base}_f_2"
            elif '_f_1' in feat or 'f_1_' in feat:
                if feat.startswith('f_1_'):
                    f1_col = feat
                    f2_col = feat.replace('f_1_', 'f_2_')
                else:
                    f1_col = feat
                    f2_col = feat.replace('_f_1', '_f_2')
                if f2_col not in all_cols: f1_col = None
            elif '_f_2' in feat or 'f_2_' in feat:
                if feat.startswith('f_2_'):
                    f2_col = feat
                    f1_col = feat.replace('f_2_', 'f_1_')
                else:
                    f2_col = feat
                    f1_col = feat.replace('_f_2', '_f_1')
                if f1_col not in all_cols: f1_col = None
                
            if f1_col and f2_col:
                if (f1_col, f2_col) not in pairs:
                    pairs.add((f1_col, f2_col))

        # CRITICAL: Sort pairs to ensure deterministic order
        pairs = sorted(list(pairs))
        return pairs

    def _get_feature_pairs(self, feature_cols):
        return self._get_feature_pairs_robust(feature_cols)

    def build_history(self):
        print("Building Master 3 History Buffer...")
        data_path = os.path.join(self.base_dir, 'data/training_data_enhanced.csv')
        if not os.path.exists(data_path):
            print("Error: Enhanced data not found. Cannot build history.")
            self.history = {}
            self.latest_stats = {}
            return

        df = pd.read_csv(data_path)
        df['event_date'] = pd.to_datetime(df['event_date'])
        df = df.sort_values('event_date')
        
        # 1. Pre-compute data matrix for sequences
        # (N, 2, D)
        data_matrix = np.zeros((len(df), 2, self.seq_input_dim))
        for i, (c1, c2) in enumerate(self.siamese_pairs):
            data_matrix[:, 0, i] = df[c1].fillna(0).values
            data_matrix[:, 1, i] = df[c2].fillna(0).values
            
        # 2. Build History: Fighter -> List of vectors (latest last)
        self.history = {} # fighter -> list of numpy arrays (D,)
        self.latest_stats = {} # fighter -> dict of latest static stats
        
        f1_names = df['f_1_name'].values
        f2_names = df['f_2_name'].values
        
        # Convert to records for fast access
        records = df.to_dict('records')
        
        for i in range(len(df)):
            n1 = f1_names[i]
            n2 = f2_names[i]
            row_data = records[i]
            
            # Update History
            vec1 = data_matrix[i, 0, :]
            vec2 = data_matrix[i, 1, :]
            
            if n1 not in self.history: self.history[n1] = []
            if n2 not in self.history: self.history[n2] = []
            
            self.history[n1].append(vec1)
            self.history[n2].append(vec2)
            
            # Store latest static stats
            stats1 = {}
            stats2 = {}
            
            # Dynamic Elo
            stats1['elo'] = row_data.get('dynamic_elo_f1', 1500)
            stats2['elo'] = row_data.get('dynamic_elo_f2', 1500)
            
            # Chin
            stats1['chin_score'] = row_data.get('f_1_chin_score', 5)
            stats2['chin_score'] = row_data.get('f_2_chin_score', 5)
            
            # PEAR
            stats1['pear_pace'] = row_data.get('f_1_pear_pace', 0)
            stats1['pear_lag'] = row_data.get('f_1_pear_lag', 0)
            stats2['pear_pace'] = row_data.get('f_2_pear_pace', 0)
            stats2['pear_lag'] = row_data.get('f_2_pear_lag', 0)
            
            self.latest_stats[n1] = stats1
            self.latest_stats[n2] = stats2
            
        print(f"History built for {len(self.history)} fighters.")

    def get_sequence(self, fighter_name, seq_len=5):
        if fighter_name not in self.history:
            return np.zeros((seq_len, self.seq_input_dim))
        
        hist = self.history[fighter_name]
        if len(hist) >= seq_len:
            return np.array(hist[-seq_len:])
        else:
            # Pad
            pad = seq_len - len(hist)
            return np.concatenate([np.zeros((pad, self.seq_input_dim)), np.array(hist)], axis=0)

    def generate_features(self, f1_name, f2_name, f1_odds=None, f2_odds=None):
        # 1. Build Feature Vector
        # We need to construct a single row DataFrame with all 'self.features'
        
        row = {}
        
        # Get latest stats
        s1 = self.latest_stats.get(f1_name, {})
        s2 = self.latest_stats.get(f2_name, {})
        
        # Defaults
        def get_val(stats, key, default=0):
            return stats.get(key, default)
            
        # Calculate Diffs
        row['diff_dynamic_elo'] = get_val(s1, 'elo', 1500) - get_val(s2, 'elo', 1500)
        row['diff_chin_score'] = get_val(s1, 'chin_score', 5) - get_val(s2, 'chin_score', 5)
        row['diff_pear_pace'] = get_val(s1, 'pear_pace') - get_val(s2, 'pear_pace')
        row['diff_pear_lag'] = get_val(s1, 'pear_lag') - get_val(s2, 'pear_lag')
        
        # Raw Features (We need to populate ALL features expected by the model)
        # This is tricky because the model expects ~100 features.
        # Many are raw stats (slpm, etc.) which we didn't store in 'latest_stats' fully.
        # BUT, we have them in the 'history' vectors!
        # The history vector contains the values of the features for that fighter.
        # The 'siamese_pairs' mapping tells us which index corresponds to which feature.
        
        # Let's use the LATEST history vector as the current state.
        # This is an approximation (it's the state going INTO their LAST fight), 
        # but it's the best we have without running the full feature gen pipeline on new raw data.
        
        vec1 = self.history.get(f1_name, [np.zeros(self.siamese_input_dim)])[-1]
        vec2 = self.history.get(f2_name, [np.zeros(self.siamese_input_dim)])[-1]
        
        # Map back to columns
        for i, (c1, c2) in enumerate(self.siamese_pairs):
            # c1 is F1 col, c2 is F2 col
            row[c1] = vec1[i]
            row[c2] = vec2[i]
            
        # Overwrite Diffs (since we calculated them fresh above, which is better)
        # Actually, if we use the old vector, the diffs are from the old fight.
        # We MUST recalculate diffs.
        
        # Odds
        if f1_odds is not None: row['f_1_odds'] = f1_odds
        if f2_odds is not None: row['f_2_odds'] = f2_odds
        
        # Create DF
        X = pd.DataFrame([row])
        
        # Ensure all cols exist
        for c in self.features:
            if c not in X.columns:
                X[c] = 0
                
        X = X[self.features].fillna(0)
        return X

    def predict(self, f1_name, f2_name, f1_odds, f2_odds):
        X = self.generate_features(f1_name, f2_name, f1_odds, f2_odds)
        
        # 2. XGB Prediction
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1][0]
        
        # 3. Siamese Prediction
        siamese_prob = 0.5
        if self.siamese_model:
            # Prepare Input
            # We need (1, input_dim) for f1 and f2
            # We can extract from X using siamese_pairs
            
            f1_data = []
            f2_data = []
            for c1, c2 in self.siamese_pairs:
                f1_data.append(X[c1].values[0])
                f2_data.append(X[c2].values[0])
                
            f1_arr = np.array([f1_data])
            f2_arr = np.array([f2_data])
            
            # Scale
            f1_arr = self.scaler.transform(f1_arr)
            f2_arr = self.scaler.transform(f2_arr)
            
            # Sequences
            seq1 = self.get_sequence(f1_name)
            seq2 = self.get_sequence(f2_name)
            
            # Scale Sequences? 
            # The scaler was fit on the static features.
            # The sequence features ARE the static features (just over time).
            # So yes, we should scale them.
            # seq shape: (5, D)
            # We need to reshape to (5*D) or loop? Scaler expects (N, D).
            
            seq1_scaled = self.scaler.transform(seq1)
            seq2_scaled = self.scaler.transform(seq2)
            
            # Reshape for Batch
            seq1_batch = np.array([seq1_scaled])
            seq2_batch = np.array([seq2_scaled])
            
            with torch.no_grad():
                t_f1 = torch.FloatTensor(f1_arr).to(self.device)
                t_f2 = torch.FloatTensor(f2_arr).to(self.device)
                t_s1 = torch.FloatTensor(seq1_batch).to(self.device)
                t_s2 = torch.FloatTensor(seq2_batch).to(self.device)
                
                siamese_prob = float(self.siamese_model(t_f1, t_f2, t_s1, t_s2).item())
                
        # 4. Ensemble
        w = self.params.get('ensemble_xgb_weight', 0.5)
        prob = w * xgb_prob + (1 - w) * siamese_prob
        
        return prob, xgb_prob, siamese_prob
