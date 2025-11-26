import pandas as pd
import numpy as np
import joblib
import json
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
from models import SiameseMatchupNet, prepare_siamese_data

def calculate_ece(probs, y_true, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
        bin_count = np.sum(bin_mask)
        
        if bin_count > 0:
            prob_in_bin = probs[bin_mask]
            acc_in_bin = y_true[bin_mask].mean()
            avg_prob = prob_in_bin.mean()
            ece += (bin_count / len(probs)) * np.abs(acc_in_bin - avg_prob)
            
    return ece

def compare_calibration():
    print("=== Comparing Calibration Methods ===")
    
    # 1. Load Data
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Calibration Set: 2023-2024
    mask_calib = (df['event_date'].dt.year.isin([2023, 2024]))
    # Test Set: 2025
    mask_test = (df['event_date'].dt.year == 2025) & (df['f_1_odds'] > 1.0) & (df['f_2_odds'] > 1.0)
    
    df_calib = df[mask_calib].copy()
    df_test = df[mask_test].copy()
    
    print(f"Calibration Samples: {len(df_calib)}")
    print(f"Test Samples: {len(df_test)}")
    
    y_calib = df_calib['target'].values
    y_test = df_test['target'].values
    
    # 2. Load Models
    xgb_model = joblib.load('models/xgb_optimized.pkl')
    scaler = joblib.load('models/siamese_scaler.pkl')
    with open('features.json', 'r') as f: features = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('models/siamese_optimized.pth', map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    # 3. Get Raw Component Probabilities
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
    xgb_test, siam_test = get_component_probs(df_test)
    
    # Raw Ensemble (Baseline)
    w = 0.405
    ens_calib = w * xgb_calib + (1 - w) * siam_calib
    ens_test = w * xgb_test + (1 - w) * siam_test
    
    results = []
    
    # Method 1: Baseline (No Calibration)
    results.append({
        "Method": "Uncalibrated",
        "Brier": brier_score_loss(y_test, ens_test),
        "LogLoss": log_loss(y_test, ens_test),
        "ECE": calculate_ece(ens_test, y_test)
    })
    
    # Method 2: Platt Scaling on Ensemble
    lr = LogisticRegression(C=1.0, solver='lbfgs')
    lr.fit(ens_calib.reshape(-1, 1), y_calib)
    p_platt = lr.predict_proba(ens_test.reshape(-1, 1))[:, 1]
    
    results.append({
        "Method": "Platt on Ensemble",
        "Brier": brier_score_loss(y_test, p_platt),
        "LogLoss": log_loss(y_test, p_platt),
        "ECE": calculate_ece(p_platt, y_test)
    })
    
    # Method 3: Isotonic Regression on Ensemble
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(ens_calib, y_calib)
    p_iso = iso.predict(ens_test)
    
    results.append({
        "Method": "Isotonic on Ensemble",
        "Brier": brier_score_loss(y_test, p_iso),
        "LogLoss": log_loss(y_test, p_iso),
        "ECE": calculate_ece(p_iso, y_test)
    })
    
    # Method 4: Separate Calibration (Platt)
    lr_xgb = LogisticRegression(C=1.0, solver='lbfgs')
    lr_xgb.fit(xgb_calib.reshape(-1, 1), y_calib)
    
    lr_siam = LogisticRegression(C=1.0, solver='lbfgs')
    lr_siam.fit(siam_calib.reshape(-1, 1), y_calib)
    
    p_xgb_cal = lr_xgb.predict_proba(xgb_test.reshape(-1, 1))[:, 1]
    p_siam_cal = lr_siam.predict_proba(siam_test.reshape(-1, 1))[:, 1]
    
    p_sep_platt = w * p_xgb_cal + (1 - w) * p_siam_cal
    
    results.append({
        "Method": "Separate Platt",
        "Brier": brier_score_loss(y_test, p_sep_platt),
        "LogLoss": log_loss(y_test, p_sep_platt),
        "ECE": calculate_ece(p_sep_platt, y_test)
    })
    
    # Method 5: Separate Calibration (Isotonic)
    iso_xgb = IsotonicRegression(out_of_bounds='clip')
    iso_xgb.fit(xgb_calib, y_calib)
    
    iso_siam = IsotonicRegression(out_of_bounds='clip')
    iso_siam.fit(siam_calib, y_calib)
    
    p_xgb_iso = iso_xgb.predict(xgb_test)
    p_siam_iso = iso_siam.predict(siam_test)
    
    p_sep_iso = w * p_xgb_iso + (1 - w) * p_siam_iso
    
    results.append({
        "Method": "Separate Isotonic",
        "Brier": brier_score_loss(y_test, p_sep_iso),
        "LogLoss": log_loss(y_test, p_sep_iso),
        "ECE": calculate_ece(p_sep_iso, y_test)
    })
    
    # Print Results
    print(f"\n{'Method':<25} | {'Brier (Lower is Better)':<25} | {'ECE (Lower is Better)':<25}")
    print("-" * 80)
    for res in results:
        print(f"{res['Method']:<25} | {res['Brier']:<25.5f} | {res['ECE']:<25.5f}")

if __name__ == "__main__":
    compare_calibration()
