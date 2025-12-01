import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

class UnifiedPredictor:
    def __init__(self, base_dir='master_props_2/xgb_unified'):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, 'models')
        self.load_resources()
        
    def load_resources(self):
        print("Loading Unified XGB resources...")
        
        # Load Models
        self.model_win = joblib.load(os.path.join(self.models_dir, 'xgb_unified_winner.pkl'))
        self.model_finish = joblib.load(os.path.join(self.models_dir, 'xgb_unified_finish.pkl'))
        self.model_method = joblib.load(os.path.join(self.models_dir, 'xgb_unified_method.pkl'))
        self.model_round = joblib.load(os.path.join(self.models_dir, 'xgb_unified_round.pkl'))
        
        # Load Features (from master_3)
        # We assume the script is running from repo root, so path is relative
        # But we should be robust.
        # master_3 is parallel to master_props_2
        # If base_dir is 'master_props_2/xgb_unified', then master_3 is '../../master_3'
        
        # Try absolute path resolution
        # Assuming base_dir is relative to CWD
        root_dir = os.path.abspath(os.path.join(self.base_dir, '..', '..'))
        feat_path = os.path.join(root_dir, 'master_3', 'features_enhanced.json')
        
        if not os.path.exists(feat_path):
            # Fallback for different CWD
            feat_path = 'master_3/features_enhanced.json'
            
        with open(feat_path, 'r') as f:
            self.features = json.load(f)
            
        print(f"Loaded {len(self.features)} features.")
        
    def predict(self, fighter1_name, fighter2_name, features_df=None):
        """
        Predicts Winner and Props for a matchup.
        features_df: DataFrame containing the pre-calculated features for the matchup.
                     If None, we would need to generate them (not implemented here for simplicity, 
                     assuming API handles feature generation via master_3 utils or similar).
        """
        if features_df is None:
            raise ValueError("features_df must be provided")
            
        # Ensure features match
        X = features_df[self.features].fillna(0)
        
        # Winner
        p_win = self.model_win.predict_proba(X)[0, 1]
        
        # Props
        p_finish = self.model_finish.predict_proba(X)[0, 1]
        p_method = self.model_method.predict_proba(X)[0] # [KO, Sub]
        p_round = self.model_round.predict_proba(X)[0]   # [R1..R5]
        
        # Logic
        if p_win > 0.5:
            winner = fighter1_name
            conf = p_win
        else:
            winner = fighter2_name
            conf = 1 - p_win
            
        # Method
        prob_ko = p_finish * p_method[0]
        prob_sub = p_finish * p_method[1]
        prob_dec = 1 - p_finish
        
        methods = {'KO/TKO': prob_ko, 'Submission': prob_sub, 'Decision': prob_dec}
        pred_method = max(methods, key=methods.get)
        
        # Round
        best_rnd_idx = np.argmax(p_round)
        pred_round = best_rnd_idx + 1
        
        return {
            'winner': winner,
            'confidence': float(conf),
            'method': pred_method,
            'round': int(pred_round),
            'probabilities': {
                'win': float(p_win),
                'finish': float(p_finish),
                'method': methods,
                'round': p_round.tolist()
            }
        }
