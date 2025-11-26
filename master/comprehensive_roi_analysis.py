import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import torch
from sklearn.metrics import accuracy_score
from models import SiameseMatchupNet, prepare_siamese_data

# --- Configuration ---
KELLY_FRACTIONS = [1.0, 0.5, 0.25, 0.125, 0.0625] # Full, Half, Quarter, Eighth, Sixteenth
YEARS = [2024, 2025]
MODELS = {
    "With Odds": {
        "xgb_path": "models/xgb_optimized.pkl",
        "siamese_path": "models/siamese_optimized.pth",
        "scaler_path": "models/siamese_scaler.pkl",
        "cols_path": "models/siamese_cols.json",
        "meta_path": "models/model_metadata.json",
        "features_file": "features.json" # Original features with odds
    },
    "No Odds": {
        "xgb_path": "models/no_odds/xgb_no_odds.pkl",
        "siamese_path": "models/no_odds/siamese_no_odds.pth",
        "scaler_path": "models/no_odds/siamese_scaler.pkl",
        "cols_path": "models/no_odds/siamese_cols.json",
        "meta_path": "models/model_metadata.json", # Assuming weight is similar or we can recalculate
        "features_file": "features.json" # We will filter this dynamically
    }
}

def load_model_pipeline(model_config, model_type):
    print(f"Loading {model_type} pipeline...")
    
    # Load XGBoost
    xgb_model = joblib.load(model_config['xgb_path'])
    
    # Load Features
    with open(model_config['features_file'], 'r') as f:
        all_features = json.load(f)
        
    if model_type == "No Odds":
        odds_keywords = ['odds', 'implied_prob']
        features = [f for f in all_features if not any(k in f.lower() for k in odds_keywords)]
    else:
        features = all_features
        
    # Load Scaler & Cols
    scaler = joblib.load(model_config['scaler_path'])
    with open(model_config['cols_path'], 'r') as f:
        siamese_cols = json.load(f)
        
    # Load Siamese
    # We need input_dim to init model
    # Hack: Assume hidden_dim=64 from training
    input_dim = len(siamese_cols) # This might be slightly off if cols logic changed, but usually correct
    
    # Actually, let's just use the saved state dict. 
    # We need to know the input dimension the model was trained with.
    # In prepare_siamese_data, input_dim is len(f1_feats).
    # We can infer it from the state dict 'encoder.0.weight' shape [64, input_dim]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_config['siamese_path'], map_location=device)
    input_dim = state_dict['encoder.0.weight'].shape[1]
    
    siamese_model = SiameseMatchupNet(input_dim, hidden_dim=64).to(device)
    siamese_model.load_state_dict(state_dict)
    siamese_model.eval()
    
    return xgb_model, siamese_model, scaler, features, device

def get_predictions(df, xgb_model, siamese_model, scaler, features, device):
    # Prepare X
    X = df[[c for c in features if c in df.columns]].fillna(0)
    
    # XGB Preds
    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    
    # Siamese Preds
    f1_data, f2_data, _, _ = prepare_siamese_data(X, features)
    
    # Scale
    f1_data = scaler.transform(f1_data)
    f2_data = scaler.transform(f2_data)
    
    t_f1 = torch.FloatTensor(f1_data).to(device)
    t_f2 = torch.FloatTensor(f2_data).to(device)
    
    with torch.no_grad():
        siamese_probs = siamese_model(t_f1, t_f2).cpu().numpy()
        
    # Ensemble (Fixed weight 0.4 for now, or read from metadata)
    w = 0.405
    ens_probs = w * xgb_probs + (1 - w) * siamese_probs
    
    return ens_probs

def run_simulation(df, probs, year, model_name):
    print(f"\n--- Simulation: {year} | {model_name} ---")
    
    # Filter for valid odds for betting simulation (even if model didn't use them)
    # We can't bet if we don't know the odds!
    valid_betting_df = df[(df['f_1_odds'] > 1.0) & (df['f_2_odds'] > 1.0)].copy()
    valid_indices = valid_betting_df.index
    
    # Align probs
    # We need to map df index to probs array index
    # probs array corresponds to original df order
    # Create a series for easy mapping
    prob_series = pd.Series(probs, index=df.index)
    valid_probs = prob_series[valid_indices].values
    
    print(f"Fights with valid odds: {len(valid_betting_df)}")
    
    # Calculate Accuracy on this set
    preds = (valid_probs > 0.5).astype(int)
    targets = valid_betting_df['target'].values
    acc = accuracy_score(targets, preds)
    print(f"Model Accuracy: {acc:.2%}")
    
    results_table = []
    
    for fraction in KELLY_FRACTIONS:
        bankroll = 1000.0
        wagered = 0.0
        profit = 0.0
        
        for i, (prob, target) in enumerate(zip(valid_probs, targets)):
            row = valid_betting_df.iloc[i]
            
            # Decide bet
            pred_class = 1 if prob > 0.5 else 0
            my_prob = prob if pred_class == 1 else 1 - prob
            
            if pred_class == 1:
                odds = row['f_1_odds']
                won = (target == 1)
            else:
                odds = row['f_2_odds']
                won = (target == 0)
                
            # Kelly Calc
            b = odds - 1.0
            q = 1.0 - my_prob
            f_star = (b * my_prob - q) / b
            
            if f_star > 0:
                stake = bankroll * f_star * fraction
                # Cap at 20%
                max_stake = bankroll * 0.20
                if stake > max_stake: stake = max_stake
                
                wagered += stake
                
                if won:
                    change = stake * (odds - 1)
                    bankroll += change
                    profit += change
                else:
                    change = -stake
                    bankroll += change
                    profit -= stake
        
        # Metrics
        roi = (profit / wagered * 100) if wagered > 0 else 0.0
        total_return_pct = ((bankroll - 1000) / 1000) * 100
        
        results_table.append({
            "Fraction": f"Kelly {fraction}",
            "ROI": roi,
            "Total Return": total_return_pct,
            "Final Bankroll": bankroll
        })
        
    # Print Table
    print(f"{'Strategy':<15} | {'ROI':<8} | {'Total Return':<12} | {'Final Bankroll':<15}")
    print("-" * 60)
    for res in results_table:
        print(f"{res['Fraction']:<15} | {res['ROI']:>6.2f}% | {res['Total Return']:>10.1f}% | ${res['Final Bankroll']:>10.2f}")

def main():
    print("Loading Data...")
    df = pd.read_csv('data/training_data.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['Year'] = df['event_date'].dt.year
    
    for model_name, config in MODELS.items():
        try:
            xgb_m, siamese_m, scaler, feats, device = load_model_pipeline(config, model_name)
        except Exception as e:
            print(f"Skipping {model_name}: {e}")
            continue
            
        # Get predictions for all relevant years
        # We process year by year to simulate "resetting" bankroll or just analysis
        
        for year in YEARS:
            year_df = df[df['Year'] == year].copy()
            if len(year_df) == 0:
                print(f"No data for {year}")
                continue
                
            probs = get_predictions(year_df, xgb_m, siamese_m, scaler, feats, device)
            run_simulation(year_df, probs, year, model_name)

if __name__ == "__main__":
    main()
