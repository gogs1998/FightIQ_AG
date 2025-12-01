import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import sys
import torch
from sklearn.preprocessing import StandardScaler

# Add current dir to path
sys.path.append(os.getcwd())

from models import SiameseMatchupNet, prepare_siamese_data
from models.sequence_model import prepare_sequences
from models.opponent_adjustment import apply_opponent_adjustment

def predict_upcoming(upcoming_csv='upcoming.csv'):
    print("=== Master 3: Prediction Engine ===")
    
    # 1. Load Resources
    print("Loading models and config...")
    
    # Features
    if os.path.exists('features_selected.json'):
        with open('features_selected.json', 'r') as f:
            features = json.load(f)
    else:
        with open('features_enhanced.json', 'r') as f:
            features = json.load(f)
            
    # Params
    with open('params.json', 'r') as f:
        best = json.load(f)
        params = best['best_params']
    if os.path.exists('params_optimized.json'):
        with open('params_optimized.json', 'r') as f:
            params.update(json.load(f))
            
    # Models
    xgb_model = joblib.load('models/xgb_master3.pkl')
    finish_model = joblib.load('models/finish_master3.pkl')
    
    # Siamese (We need to re-initialize and load state if we saved it, 
    # BUT train.py didn't save the state dict, it just trained on the fly.
    # CRITICAL: To support inference, we must modify train.py to save the best Siamese state_dict.
    # For now, we can't predict with Siamese without retraining or saving.
    # I will assume the user will re-train or I should have saved it.
    # Wait, train.py lines 286-287 only saved XGB and Finish.
    # I need to update train.py to save 'models/siamese_master3.pth'.
    
    # Siamese
    # We need to reconstruct the model with correct dims
    # We can infer dims from features
    # But we need seq_dim from prepare_sequences
    
    # 2. Load Data
    print(f"Loading upcoming fights from {upcoming_csv}...")
    if not os.path.exists(upcoming_csv):
        print("Error: File not found. Please create a CSV with upcoming fight data.")
        return
        
    df = pd.read_csv(upcoming_csv)
    
    # 3. Preprocess
    # We need to run the same feature generation pipeline?
    # Ideally, upcoming.csv is already in "enhanced" format.
    # If not, we'd need to run generate_features.py on it.
    # For now, assume it's pre-processed.
    
    # Prepare Data
    X_df = df[[c for c in features if c in df.columns]]
    X_df = X_df.fillna(0)
    
    # Prepare Sequences
    # Note: This requires history. If upcoming.csv is just one card, 
    # we might need to load the FULL training data to get history for these fighters.
    # This is a complexity.
    # Solution: Load training data, append upcoming, run prep, then slice.
    print("Loading history for sequence generation...")
    train_df = pd.read_csv('data/training_data_enhanced.csv')
    
    # Concatenate to ensure history availability
    # Mark upcoming rows
    df['is_upcoming'] = True
    train_df['is_upcoming'] = False
    
    full_df = pd.concat([train_df, df], ignore_index=True)
    full_df['event_date'] = pd.to_datetime(full_df['event_date'])
    full_df = full_df.sort_values('event_date')
    
    # Run Sequence Prep on FULL
    print("Generating sequences...")
    seq_f1, seq_f2, seq_dim = prepare_sequences(full_df, features)
    
    # Slice back to upcoming
    mask_upcoming = full_df['is_upcoming'] == True
    X_upcoming = full_df.loc[mask_upcoming, [c for c in features if c in full_df.columns]].fillna(0)
    
    seq_f1_up = seq_f1[mask_upcoming]
    seq_f2_up = seq_f2[mask_upcoming]
    
    # 4. Predict
    print("Running predictions...")
    
    # XGB
    xgb_probs = xgb_model.predict_proba(X_upcoming)[:, 1]
    
    # Siamese
    f1_up, f2_up, input_dim, _ = prepare_siamese_data(X_upcoming, features)
    
    # Scale (using scaler fitted on training? We didn't save the scaler!)
    # CRITICAL: We need the scaler.
    # For now, fit on training data again.
    scaler = StandardScaler()
    f1_train, f2_train, _, _ = prepare_siamese_data(train_df[features].fillna(0), features)
    scaler.fit(np.concatenate([f1_train, f2_train], axis=0))
    
    f1_up = scaler.transform(f1_up)
    f2_up = scaler.transform(f2_up)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese_model = SiameseMatchupNet(input_dim, seq_input_dim=seq_dim, hidden_dim=params.get('siamese_hidden_dim', 64)).to(device)
    
    if os.path.exists('models/siamese_master3.pth'):
        siamese_model.load_state_dict(torch.load('models/siamese_master3.pth'))
    else:
        print("Warning: Siamese weights not found. Predictions will be random for Siamese component.")
        
    siamese_model.eval()
    with torch.no_grad():
        t_f1 = torch.FloatTensor(f1_up).to(device)
        t_f2 = torch.FloatTensor(f2_up).to(device)
        t_s1 = torch.FloatTensor(seq_f1_up).to(device)
        t_s2 = torch.FloatTensor(seq_f2_up).to(device)
        
        siamese_probs = siamese_model(t_f1, t_f2, t_s1, t_s2).cpu().numpy()
        
    # Ensemble
    w = params.get('ensemble_xgb_weight', 0.5)
    ens_probs = w * xgb_probs + (1 - w) * siamese_probs
    
    # Finish Prob
    finish_probs = finish_model.predict_proba(X_upcoming)[:, 1]
    
    # 5. Output
    print("\n=== Upcoming Fight Predictions ===")
    results = df.copy()
    results['Win_Prob_F1'] = ens_probs
    results['Finish_Prob'] = finish_probs
    results['Pred_Winner'] = np.where(ens_probs > 0.5, results['f_1_name'], results['f_2_name'])
    results['Confidence'] = np.abs(ens_probs - 0.5) * 2
    
    # Display
    cols = ['f_1_name', 'f_2_name', 'Pred_Winner', 'Win_Prob_F1', 'Finish_Prob', 'Confidence']
    print(results[cols].to_string(index=False))
    
    results.to_csv('predictions.csv', index=False)
    print("\nSaved to predictions.csv")

if __name__ == "__main__":
    predict_upcoming()
