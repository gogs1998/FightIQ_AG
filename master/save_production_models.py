import pandas as pd
import numpy as np
import joblib
import json
import torch
from sklearn.isotonic import IsotonicRegression
from models import SiameseMatchupNet, prepare_siamese_data

def save_production_models():
    print("=== Training & Saving Production Isotonic Models ===")
    
    # 1. Load Data
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Use 2023-2024 for Calibration (The same set we verified on)
    # Ideally, we should use as much recent data as possible without leaking into the test set.
    # Since we are deploying for the FUTURE, we can use 2023-2025 data to calibrate?
    # No, let's stick to the verified 2023-2024 set to be safe and consistent with the paper.
    # Or better: Use 2023-2025 (up to today) to calibrate for tomorrow.
    # Let's use 2023-2025.
    
    mask_calib = (df['event_date'].dt.year >= 2023)
    df_calib = df[mask_calib].copy()
    y_calib = df_calib['target'].values
    
    print(f"Calibration Set (2023-Present): {len(df_calib)} fights")
    
    # 2. Load Base Models
    xgb_model = joblib.load('models/xgb_optimized.pkl')
    scaler = joblib.load('models/siamese_scaler.pkl')
    with open('features.json', 'r') as f: features = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('models/siamese_optimized.pth', map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    # 3. Get Component Probs
    def get_component_probs(dframe):
        X = dframe[[c for c in features if c in dframe.columns]].fillna(0)
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        
        f1, f2, _, _ = prepare_siamese_data(X, features)
        f1 = scaler.transform(f1)
        f2 = scaler.transform(f2)
        t1 = torch.FloatTensor(f1).to(device)
        t2 = torch.FloatTensor(f2).to(device)
        with torch.no_grad():
            p_siam = siamese_model(t1, t2).cpu().numpy()
        return p_xgb, p_siam

    xgb_calib, siam_calib = get_component_probs(df_calib)
    
    # 4. Train Isotonic Calibrators
    iso_xgb = IsotonicRegression(out_of_bounds='clip')
    iso_xgb.fit(xgb_calib, y_calib)
    
    iso_siam = IsotonicRegression(out_of_bounds='clip')
    iso_siam.fit(siam_calib, y_calib)
    
    # 5. Save
    joblib.dump(iso_xgb, 'models/iso_xgb.pkl')
    joblib.dump(iso_siam, 'models/iso_siam.pkl')
    
    print("Saved models/iso_xgb.pkl and models/iso_siam.pkl")

if __name__ == "__main__":
    save_production_models()
