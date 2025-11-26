from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np
import uvicorn
import os
import sys

# Add v2 to path
sys.path.append(os.path.join(os.getcwd(), 'v2'))
from models.analyst_ensemble import AnalystEnsemble
from models.gambler_model import GamblerModel
from api_support.matchup_engine import MatchupEngine

app = FastAPI(title="FightIQ v2 API", description="UFC Prediction Engine (Analyst + Gambler)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Resources
engine = None
xgb_model = None
siamese_model = None
siamese_scaler = None
siamese_cols = []
xgb_weight = 0.5

class FightRequest(BaseModel):
    f1_name: str
    f2_name: str
    f1_odds: float
    f2_odds: float

class PredictionResponse(BaseModel):
    winner: str
    confidence: float
    f1_prob: float
    f2_prob: float
    is_value: bool
    bet_target: str
    edge: float
    method_probs: dict = {} # Placeholder for future
    conformal_set: list = [] # [0, 1] or [1] etc.

@app.on_event("startup")
def load_resources():
    global engine, xgb_model, siamese_model, siamese_scaler, siamese_cols, xgb_weight
    print("Loading v2 resources...")
    
    # 1. Matchup Engine
    engine = MatchupEngine()
    
    # Load Models
    try:
        print("Loading optimized models...")
        xgb_model = joblib.load('v2/models/xgb_optimized.pkl')
        
        # Load Siamese
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load scaler
        siamese_scaler = joblib.load('v2/models/siamese_scaler.pkl')
        
        # Load Siamese columns
        with open('v2/models/siamese_cols.json', 'r') as f:
            siamese_cols = json.load(f)
            
        # Initialize Siamese Model (Hidden dim 64 from Optuna)
        siamese_model = SiameseMatchupNet(len(siamese_cols), hidden_dim=64).to(device)
        siamese_model.load_state_dict(torch.load('v2/models/siamese_optimized.pth'))
        siamese_model.eval()
        
        # Load Metadata for weights
        with open('v2/models/model_metadata.json', 'r') as f:
            meta = json.load(f)
            xgb_weight = meta['xgb_weight']
            
        print(f"Models loaded. Ensemble Weight: XGB={xgb_weight:.2f}, Siamese={1-xgb_weight:.2f}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Falling back to legacy models...")
        # Fallback logic or exit
        xgb_model = None
        siamese_model = None

@app.post("/predict", response_model=PredictionResponse)
def predict(fight: FightRequest):
    if not engine or not xgb_model:
        raise HTTPException(status_code=503, detail="Models not loaded.")
        
    f1 = fight.f1_name
    f2 = fight.f2_name
    
    # 1. Build Matchup Features
    try:
        X = engine.build_matchup(f1, f2, fight.f1_odds, fight.f2_odds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error building matchup: {e}")
        
    # 2. Generate Prediction
    try:
        # XGBoost Prediction
        # Ensure X has correct columns for XGB
        # The model expects specific columns. MatchupEngine returns a DataFrame with many cols.
        # XGBoost handles extra columns if feature_names match? No, usually need to filter.
        # But our trained model has feature_names_in_
        
        # Filter X to model features
        model_feats = xgb_model.get_booster().feature_names
        X_xgb = X[model_feats]
        xgb_prob = float(xgb_model.predict_proba(X_xgb)[:, 1][0])
        
        # Siamese Prediction
        siamese_prob = 0.5 # Default
        if siamese_model:
            # Prepare Siamese Data
            # We need to extract the specific columns expected by Siamese
            # We loaded siamese_cols in startup
            # And we have siamese_scaler
            
            # Extract raw values
            # X has 1 row
            # We need to reconstruct f1_data and f2_data based on siamese_cols logic?
            # Actually, siamese_cols is a list of (c1, c2) tuples? No, it's a list of f1_feats.
            # Wait, prepare_siamese_data returns f1_feats.
            # And we saved siamese_cols = f1_feats.
            
            # We need to find the corresponding f2_feats.
            # The logic was:
            # f1_col = feat
            # f2_col = replace f1 with f2
            
            f1_data = []
            f2_data = []
            
            for f1_col in siamese_cols:
                # Reconstruct f2_col name
                if f1_col.startswith('f_1_'):
                    f2_col = f1_col.replace('f_1_', 'f_2_')
                elif '_f_1' in f1_col:
                    f2_col = f1_col.replace('_f_1', '_f_2')
                else:
                    # Should not happen given our logic, but fallback
                    f2_col = f1_col 
                
                v1 = X[f1_col].values[0] if f1_col in X.columns else 0
                v2 = X[f2_col].values[0] if f2_col in X.columns else 0
                
                f1_data.append(v1)
                f2_data.append(v2)
            
            f1_arr = np.array([f1_data])
            f2_arr = np.array([f2_data])
            
            # Scale
            # We fit scaler on concatenated data.
            f1_arr = siamese_scaler.transform(f1_arr)
            f2_arr = siamese_scaler.transform(f2_arr)
            
            # Predict
            t_f1 = torch.FloatTensor(f1_arr).to(device)
            t_f2 = torch.FloatTensor(f2_arr).to(device)
            siamese_prob = float(siamese_model(t_f1, t_f2).item())
            
        # Ensemble
        # Load weight from metadata? We did in startup.
        # Global xgb_weight
        p_f1 = xgb_weight * xgb_prob + (1 - xgb_weight) * siamese_prob
        p_f2 = 1.0 - p_f1
        
        # Conformal Set (Simplified for now)
        c_set = []
        if p_f1 > 0.2: c_set.append(f1)
        if p_f2 > 0.2: c_set.append(f2)
        
    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model failed: {e}")

    # 3. Betting Recommendation (Simplified)
    is_value = False
    bet_target = ""
    edge = 0.0
    
    if p_f1 > 0.5:
        implied = 1 / fight.f1_odds
        if p_f1 > implied * 1.05: # 5% margin
            is_value = True
            bet_target = f1
            edge = p_f1 - implied
    else:
        implied = 1 / fight.f2_odds
        if p_f2 > implied * 1.05:
            is_value = True
            bet_target = f2
            edge = p_f2 - implied

    winner = f1 if p_f1 > 0.5 else f2
    confidence = max(p_f1, p_f2)
    
    return {
        "winner": winner,
        "confidence": confidence,
        "f1_prob": p_f1,
        "f2_prob": p_f2,
        "is_value": is_value,
        "bet_target": bet_target,
        "edge": edge,
        "conformal_set": c_set
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
