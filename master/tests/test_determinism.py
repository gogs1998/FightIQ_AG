import pytest
import pandas as pd
import numpy as np
import joblib
import json
import torch
import os
import sys

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import SiameseMatchupNet

# --- Fixtures ---
@pytest.fixture(scope="session")
def resources():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    res = {}
    res['xgb'] = joblib.load(os.path.join(base_dir, 'models', 'xgb_optimized.pkl'))
    res['iso_xgb'] = joblib.load(os.path.join(base_dir, 'models', 'iso_xgb.pkl'))
    res['iso_siam'] = joblib.load(os.path.join(base_dir, 'models', 'iso_siam.pkl'))
    res['scaler'] = joblib.load(os.path.join(base_dir, 'models', 'siamese_scaler.pkl'))
    
    with open(os.path.join(base_dir, 'features.json'), 'r') as f:
        res['features'] = json.load(f)
    with open(os.path.join(base_dir, 'models', 'siamese_cols.json'), 'r') as f:
        res['siamese_cols'] = json.load(f)
        
    device = torch.device('cpu')
    state_dict = torch.load(os.path.join(base_dir, 'models', 'siamese_optimized.pth'), map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    res['siamese_model'] = model
    
    return res

@pytest.fixture(scope="session")
def golden_refs():
    path = os.path.join(os.path.dirname(__file__), 'data', 'golden_refs.json')
    with open(path, 'r') as f:
        return json.load(f)

# --- Tests ---

def test_feature_determinism(golden_refs, resources):
    """
    Assert that constructing features from raw stats produces the EXACT same feature vector.
    This catches drift in feature engineering logic.
    """
    for ref in golden_refs:
        scenario = ref['scenario']
        expected_features = ref['features']
        
        s1 = ref['inputs']['s1']
        s2 = ref['inputs']['s2']
        f1_odds = scenario['f1_odds']
        f2_odds = scenario['f2_odds']
        feature_names = resources['features']
        
        row = {}
        for k, v in s1.items(): row[f'f_1_{k}'] = v
        for k, v in s2.items(): row[f'f_2_{k}'] = v
        row['f_1_odds'] = f1_odds
        row['f_2_odds'] = f2_odds
        row['diff_odds'] = f1_odds - f2_odds
        
        for feat in feature_names:
            if feat in row: continue
            val = 0.0
            if feat.startswith('diff_'):
                base = feat[5:]
                v1 = s1.get(base)
                v2 = s2.get(base)
                if v1 is not None and v2 is not None: val = v1 - v2
            row[feat] = val
        for feat in feature_names:
            if feat not in row: row[feat] = 0.0
            
        # Assertions
        for feat, expected_val in expected_features.items():
            actual_val = row.get(feat, 0.0)
            
            # Handle None/NaN
            if pd.isna(expected_val) and pd.isna(actual_val):
                continue # Both are NaN, so they match
                
            if actual_val is None: actual_val = 0.0
            if expected_val is None: expected_val = 0.0
            
            # Handle float comparison
            if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                assert abs(actual_val - expected_val) < 1e-6, f"Feature mismatch: {feat} (Expected {expected_val}, Got {actual_val})"
            else:
                assert actual_val == expected_val, f"Feature mismatch: {feat}"

def test_model_determinism(golden_refs, resources):
    """
    Assert that the models produce the EXACT same probabilities given the same inputs.
    This catches accidental model overwrites or environment changes (e.g. sklearn version).
    """
    for ref in golden_refs:
        expected_out = ref['outputs']
        features_dict = ref['features']
        
        X = pd.DataFrame([features_dict])
        
        # XGB
        X_xgb = X[[c for c in resources['features'] if c in X.columns]]
        p_xgb_raw = float(resources['xgb'].predict_proba(X_xgb)[:, 1][0])
        p_xgb_iso = float(resources['iso_xgb'].predict([p_xgb_raw])[0])
        
        assert abs(p_xgb_raw - expected_out['p_xgb_raw']) < 1e-6, "XGB Raw mismatch"
        assert abs(p_xgb_iso - expected_out['p_xgb_iso']) < 1e-6, "XGB Iso mismatch"
        
        # Siamese
        f1_vec = []
        f2_vec = []
        for col in resources['siamese_cols']:
            val1 = 0.0
            if col in X.columns: val1 = X[col].values[0]
            f1_vec.append(val1)
            col2 = None
            if col.startswith('f_1_'): col2 = col.replace('f_1_', 'f_2_')
            elif '_f_1' in col: col2 = col.replace('_f_1', '_f_2')
            val2 = 0.0
            if col2 and col2 in X.columns: val2 = X[col2].values[0]
            f2_vec.append(val2)
            
        f1_feat = resources['scaler'].transform(np.array([f1_vec]))
        f2_feat = resources['scaler'].transform(np.array([f2_vec]))
        
        t1 = torch.FloatTensor(f1_feat)
        t2 = torch.FloatTensor(f2_feat)
        
        with torch.no_grad():
            p_siam_raw = resources['siamese_model'](t1, t2).item()
            
        assert abs(p_siam_raw - expected_out['p_siam_raw']) < 1e-5, "Siamese Raw mismatch"
