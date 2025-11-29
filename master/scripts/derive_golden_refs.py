import pandas as pd
import numpy as np
import joblib
import json
import torch
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import SiameseMatchupNet, prepare_siamese_data

def derive_golden_refs():
    print("=== Deriving Golden Reference Data ===")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DATA_DIR = os.path.join(BASE_DIR, 'tests', 'data')
    
    # Load Resources
    print("Loading production resources...")
    xgb_model = joblib.load(os.path.join(BASE_DIR, 'models', 'xgb_optimized.pkl'))
    iso_xgb = joblib.load(os.path.join(BASE_DIR, 'models', 'iso_xgb.pkl'))
    iso_siam = joblib.load(os.path.join(BASE_DIR, 'models', 'iso_siam.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'siamese_scaler.pkl'))
    
    with open(os.path.join(BASE_DIR, 'features.json'), 'r') as f: features = json.load(f)
    with open(os.path.join(BASE_DIR, 'fighter_db_production.json'), 'r') as f: fighter_db = json.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'siamese_cols.json'), 'r') as f: siamese_cols = json.load(f)
    
    device = torch.device('cpu') # Force CPU for deterministic testing
    state_dict = torch.load(os.path.join(BASE_DIR, 'models', 'siamese_optimized.pth'), map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    # Define Golden Bouts (Specific scenarios to lock in)
    # 1. Taira vs Moreno (The "Signal" case)
    # 2. A random mismatch (The "Pass" case)
    
    golden_scenarios = [
        {
            "id": "taira_vs_moreno_ufc323",
            "f1_name": "Tatsuro Taira",
            "f2_name": "Brandon Moreno",
            "f1_odds": 1.80,
            "f2_odds": 2.00
        },
        {
            "id": "jones_vs_aspinal_hypothetical",
            "f1_name": "Jon Jones",
            "f2_name": "Tom Aspinall",
            "f1_odds": 2.50,
            "f2_odds": 1.55
        }
    ]
    
    golden_results = []
    
    for scenario in golden_scenarios:
        print(f"Processing {scenario['id']}...")
        f1_name = scenario['f1_name']
        f2_name = scenario['f2_name']
        
        # Get Stats (Snapshotting the DB state)
        s1 = fighter_db.get(f1_name)
        s2 = fighter_db.get(f2_name)
        
        if not s1 or not s2:
            print(f"Warning: Could not find stats for {f1_name} or {f2_name}")
            continue
            
        # 1. Feature Construction
        row = {}
        for k, v in s1.items(): row[f'f_1_{k}'] = v
        for k, v in s2.items(): row[f'f_2_{k}'] = v
        row['f_1_odds'] = scenario['f1_odds']
        row['f_2_odds'] = scenario['f2_odds']
        row['diff_odds'] = row['f_1_odds'] - row['f_2_odds']
        
        for feat in features:
            if feat in row: continue
            val = 0.0
            if feat.startswith('diff_'):
                base = feat[5:]
                v1 = s1.get(base)
                v2 = s2.get(base)
                if v1 is not None and v2 is not None: val = v1 - v2
            row[feat] = val
        for feat in features:
            if feat not in row: row[feat] = 0.0
            
        X = pd.DataFrame([row])
        
        # 2. Model Prediction
        # XGB
        X_xgb = X[[c for c in features if c in X.columns]]
        p_xgb_raw = float(xgb_model.predict_proba(X_xgb)[:, 1][0])
        p_xgb_iso = float(iso_xgb.predict([p_xgb_raw])[0])
        
        # Siamese
        f1_vec = []
        f2_vec = []
        for col in siamese_cols:
            val1 = 0.0
            if col in X.columns: val1 = X[col].values[0]
            f1_vec.append(val1)
            col2 = None
            if col.startswith('f_1_'): col2 = col.replace('f_1_', 'f_2_')
            elif '_f_1' in col: col2 = col.replace('_f_1', '_f_2')
            val2 = 0.0
            if col2 and col2 in X.columns: val2 = X[col2].values[0]
            f2_vec.append(val2)
            
        f1_feat = scaler.transform(np.array([f1_vec]))
        f2_feat = scaler.transform(np.array([f2_vec]))
        t1 = torch.FloatTensor(f1_feat).to(device)
        t2 = torch.FloatTensor(f2_feat).to(device)
        with torch.no_grad():
            p_siam_raw = siamese_model(t1, t2).cpu().numpy()
            if np.ndim(p_siam_raw) == 0: p_siam_raw = float(p_siam_raw)
            else: p_siam_raw = p_siam_raw[0]
            
        p_siam_iso = float(iso_siam.predict([p_siam_raw])[0])
        final_prob = 0.405 * p_xgb_iso + (1 - 0.405) * p_siam_iso
        
        # Save Result
        result = {
            "scenario": scenario,
            "inputs": {
                "s1": s1,
                "s2": s2
            },
            "features": row,
            "outputs": {
                "p_xgb_raw": p_xgb_raw,
                "p_xgb_iso": p_xgb_iso,
                "p_siam_raw": float(p_siam_raw),
                "p_siam_iso": p_siam_iso,
                "final_prob": final_prob
            }
        }
        golden_results.append(result)
        
    # Write to disk
    output_path = os.path.join(TEST_DATA_DIR, 'golden_refs.json')
    with open(output_path, 'w') as f:
        json.dump(golden_results, f, indent=2)
        
    print(f"Saved {len(golden_results)} golden references to {output_path}")

if __name__ == "__main__":
    derive_golden_refs()
