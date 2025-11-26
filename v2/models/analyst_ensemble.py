import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
import os

# Ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from siamese_model import SiameseModel
from conformal import ConformalClassifier

# ... imports

class AnalystEnsemble:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.siamese_model = SiameseModel(epochs=20)
        self.conformal = ConformalClassifier(alpha=0.1) # 90% Confidence
        self.features = []
        
        # Load experimental feature list
        try:
            with open('features_elo.json', 'r') as f:
                self.experimental_features = set(json.load(f))
        except:
            print("Warning: features_elo.json not found. Using default safe list.")
            self.experimental_features = set()
        
    def preprocess(self, df, is_train=True):
        df = df.copy()
        
        # Feature Selection: Allow Experimental List + PEAR + Safe Cols
        selected_cols = []
        for c in df.columns:
            # 1. Explicitly Allowed from Experiment
            if c in self.experimental_features:
                selected_cols.append(c)
                continue
                
            # 2. PEAR Features (New)
            if 'beta_pace' in c or 'beta_lag' in c:
                selected_cols.append(c)
                continue
            
            # 3. Fallback Safe Logic (for things not in json but needed)
            if c.startswith('dynamic_elo') or c.startswith('common') or c.startswith('diff') or \
               c.endswith('_finish_rate') or c.endswith('_been_finished_rate') or c.endswith('_avg_time'):
                selected_cols.append(c)
                continue
                
            # Rolling Stats
            if any(x in c for x in ['_3_f_', '_5_f_', 'streak']):
                selected_cols.append(c)
                continue
                
            # Physical / Static
            if 'height' in c or 'reach' in c or 'age' in c or 'dob' in c or 'weight' in c:
                selected_cols.append(c)
                continue
                
            # Odds & Rankings
            if 'odds' in c or 'ranking' in c:
                selected_cols.append(c)
                continue
                
        # Filter numeric
        features = [c for c in selected_cols if df[c].dtype in [np.float64, np.int64, np.int32]]
        # Remove duplicates
        features = list(set(features))
        
        if is_train:
            self.features = features
            print(f"Selected {len(self.features)} features (Experimental + v2).")
            
        return df[self.features]

    # ... (I need to target __init__ specifically)

    def fit(self, df):
        print("Training Analyst Ensemble (XGB + Siamese)...")
        
        # 1. Train XGBoost
        print("--- Training XGBoost ---")
        X = self.preprocess(df, is_train=True)
        y = df['target']
        
        # Split for Conformal Calibration (and Siamese validation if needed)
        split_idx = int(len(df) * 0.85)
        X_train, X_cal = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_cal = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.xgb_model.fit(X_train, y_train)
        
        # 2. Train Siamese Model (Pure)
        print("--- Training Siamese Net ---")
        # Extract Top 50 features from XGBoost
        importances = self.xgb_model.feature_importances_
        feat_imp = dict(zip(self.features, importances))
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        top_features = [x[0] for x in sorted_feats[:50]]
        print(f"Top 5 features: {top_features[:5]}")
        
        # Pass TRAIN data to Siamese
        df_train = df.iloc[:split_idx]
        self.siamese_model.fit(df_train, top_features)
        
        # 3. Calibrate Conformal (on Ensemble Probs)
        print("Calibrating Conformal Prediction...")
        
        # Get predictions on Calibration Set
        df_cal = df.iloc[split_idx:]
        
        p_xgb = self.xgb_model.predict_proba(X_cal)[:, 1]
        p_sia = self.siamese_model.predict_proba(df_cal)
        
        # Average (Ensemble)
        p_ens = (p_xgb + p_sia) / 2.0
        
        self.conformal.fit(np.vstack([1-p_ens, p_ens]).T, y_cal.values)
        
        # Evaluate Train
        p_xgb_tr = self.xgb_model.predict_proba(X_train)[:, 1]
        p_sia_tr = self.siamese_model.predict_proba(df_train)
        p_ens_tr = (p_xgb_tr + p_sia_tr) / 2.0
        
        acc = accuracy_score(y_train, (p_ens_tr > 0.5).astype(int))
        ll = log_loss(y_train, p_ens_tr)
        print(f"Analyst Ensemble Train Metrics: Acc {acc:.4f}, LL {ll:.4f}")
        
    def predict(self, df):
        # XGB
        X = self.preprocess(df, is_train=False)
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        
        # Siamese
        p_sia = self.siamese_model.predict_proba(df)
        
        # Ensemble
        p_ens = (p_xgb + p_sia) / 2.0
        
        # Conformal Sets
        # predict_proba format [N, 2]
        probs_2d = np.vstack([1-p_ens, p_ens]).T
        sets = self.conformal.predict(probs_2d)
        
        return p_ens, sets

    def save(self, path='v2/models/analyst_model.pkl'):
        # We need to save both models
        # Joblib can pickle the whole class, but Siamese has PyTorch components
        # Best to save separately or use custom logic
        
        # Save XGB and Wrapper
        joblib.dump(self, path)
        
        # Save Siamese Internal State (PyTorch)
        sia_path = path.replace('.pkl', '_siamese.pt')
        self.siamese_model.save(sia_path)
        
    @classmethod
    def load(cls, path='v2/models/analyst_model.pkl'):
        obj = joblib.load(path)
        
        # Reload Siamese State
        sia_path = path.replace('.pkl', '_siamese.pt')
        if os.path.exists(sia_path):
            obj.siamese_model = SiameseModel.load(sia_path)
            
        return obj
