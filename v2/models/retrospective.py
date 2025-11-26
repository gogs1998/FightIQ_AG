import json
import numpy as np
import pandas as pd
import os

class RetrospectiveCalibrator:
    def __init__(self, fingerprints_path='experimental/data/error_fingerprints.json'):
        self.fingerprints = []
        self.load_fingerprints(fingerprints_path)
        
    def load_fingerprints(self, path):
        if not os.path.exists(path):
            print(f"Warning: Fingerprints not found at {path}. Calibration disabled.")
            return
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Parse list of [feature, dict]
        self.fingerprints = []
        for item in data:
            feat = item[0]
            stats = item[1]
            # We only care about significant Z-scores
            if abs(stats['z_score']) > 0.3:
                self.fingerprints.append({
                    'feature': feat,
                    'z_score': stats['z_score'],
                    'mean_correct': stats['mean_correct'],
                    'mean_error': stats['mean_error']
                })
        
        print(f"Loaded {len(self.fingerprints)} retrospective fingerprints.")

    def calibrate(self, probs, df):
        """
        Adjust probabilities based on error fingerprints.
        probs: np.array of shape (N,) - Probability of Class 1
        df: DataFrame containing feature values
        """
        if not self.fingerprints:
            return probs
            
        adj_probs = probs.copy()
        
        # We define a "Risk Score" for each row
        # Risk increases if feature matches the "Error" pattern
        
        # Vectorized approach is hard with dynamic features, so we iterate features
        # but vectorize over rows
        
        risk_scores = np.zeros(len(df))
        
        for fp in self.fingerprints:
            feat = fp['feature']
            if feat not in df.columns: continue
            
            vals = df[feat].fillna(0).values
            z = fp['z_score']
            mean_correct = fp['mean_correct']
            
            # If Z > 0 (Higher in Errors), then High Value = Risk
            # If Z < 0 (Lower in Errors), then Low Value = Risk
            
            # We calculate deviation from "Safe" (mean_correct)
            # If deviation is in the direction of Error, add to risk
            
            if z > 0:
                # Risk if val > mean_correct
                # Scale by Z-score
                diff = np.maximum(0, vals - mean_correct)
                risk_scores += diff * z
            else:
                # Risk if val < mean_correct
                diff = np.maximum(0, mean_correct - vals)
                risk_scores += diff * abs(z)
                
        # Normalize risk scores roughly (this is heuristic)
        # We want to penalize confidence (push towards 0.5)
        
        # Sigmoid-ish scaling for penalty
        # penalty factor between 0 and 0.2
        penalty = 0.2 * (1 - np.exp(-0.1 * risk_scores))
        
        # Apply penalty: Move towards 0.5
        # If p > 0.5, p_new = p - penalty
        # If p < 0.5, p_new = p + penalty
        
        # But don't cross 0.5? Or just dampen?
        # Let's just dampen: p_new = 0.5 + (p - 0.5) * (1 - penalty)
        
        adj_probs = 0.5 + (probs - 0.5) * (1 - penalty)
        
        return adj_probs
