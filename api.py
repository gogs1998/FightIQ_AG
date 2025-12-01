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

# Add v2 and master_props to path
sys.path.append(os.path.join(os.getcwd(), 'v2'))
sys.path.append(os.path.join(os.getcwd(), 'master_props'))
sys.path.append(os.path.join(os.getcwd(), 'master_3'))

from master_3.api_utils import Master3Predictor
from master_props_2.api_utils import UnifiedPredictor

m3_predictor = None
unified_predictor = None

# Master Props Resources
prop_model_win = None
prop_model_finish = None
prop_model_method = None
prop_model_round = None

app = FastAPI(title="FightIQ API (Unified XGB)", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    method_probs: dict = {} 
    conformal_set: list = []
    # Master Props Fields
    pred_method: str = "N/A"
    pred_round: str = "N/A"
    trifecta_prob: float = 0.0
    min_odds: str = "N/A"

@app.on_event("startup")
def load_resources():
    global m3_predictor
    global prop_model_win, prop_model_finish, prop_model_method, prop_model_round
    
    print("Loading resources...")
    
    # 1. Master 3 Predictor
    try:
        m3_predictor = Master3Predictor()
        print("Master 3 Predictor loaded.")
    except Exception as e:
        print(f"Error loading Master 3: {e}")
        import traceback
        traceback.print_exc()

    # 2. Load Master Props Models
    try:
        print("Loading Master Props models...")
        prop_model_win = joblib.load('master_props/models/production_winner.pkl')
        prop_model_finish = joblib.load('master_props/models/production_finish.pkl')
        prop_model_method = joblib.load('master_props/models/production_method.pkl')
        prop_model_round = joblib.load('master_props/models/production_round.pkl')
        print("Master Props models loaded.")
    except Exception as e:
        print(f"Error loading Master Props models: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict(fight: FightRequest):
    if not m3_predictor:
        raise HTTPException(status_code=503, detail="Master 3 Engine not loaded.")
        
    f1 = fight.f1_name
    f2 = fight.f2_name
    
    # --- Master 3 Prediction ---
    try:
        p_f1, xgb_p, sia_p = m3_predictor.predict(f1, f2, fight.f1_odds, fight.f2_odds)
        print(f"M3 Prediction: {f1}={p_f1:.1%} (XGB={xgb_p:.1%}, Sia={sia_p:.1%})")
    except Exception as e:
        print(f"M3 Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    p_f2 = 1.0 - p_f1
    winner = f1 if p_f1 > 0.5 else f2
    confidence = max(p_f1, p_f2)
    
    # Conformal Set (Approximate)
    c_set = []
    if p_f1 > 0.2: c_set.append(f1)
    if p_f2 > 0.2: c_set.append(f2)
    
    # --- Master Props Logic ---
    pred_method = "N/A"
    pred_round = "N/A"
    trifecta_prob = 0.0
    min_odds = "N/A"
    
    if prop_model_win:
        try:
            # We need features for props.
            # Master 3 uses different features than Props?
            # Props uses 'feature_utils.prepare_production_data'.
            # It expects a DataFrame with raw stats.
            # We can't easily get that from m3_predictor.
            # BUT, we can use 'feature_utils' to load the data and find the fighters?
            # Or just use the 'm3_predictor.history' to reconstruct?
            # Actually, 'prepare_production_data' loads its own data if passed a DF.
            # Let's try to use the SAME logic as predict_next_card.py:
            # It loads 'training_data_enhanced.csv' (from master_3/data) to get stats.
            
            # We can use the 'm3_predictor.latest_stats' to build a row for 'prepare_production_data'?
            # No, 'prepare_production_data' is robust.
            # Let's just create a minimal DF with names and odds, and let 'prepare_production_data' handle the lookup if it supports it.
            # Wait, 'prepare_production_data' takes a DF and calculates features.
            # It assumes the DF has the raw columns.
            
            # Hack: We will skip Props for now if we can't easily generate features, 
            # OR we rely on the fact that 'predict_next_card.py' works because it has the full CSV.
            # Here we only have names.
            
            # Solution: We need a 'PropFeatureGenerator' that looks up the fighters in the DB.
            # We can reuse 'm3_predictor's history!
            # The 'm3_predictor' has 'latest_stats'.
            # But Props model expects specific columns like 'slpm_15_f_1'.
            # 'latest_stats' has them! (I added them to keys_to_store in api_utils.py).
            
            # Let's reconstruct a row for Props.
            s1 = m3_predictor.latest_stats.get(f1, {})
            s2 = m3_predictor.latest_stats.get(f2, {})
            
            row = {}
            # We need to map 'slpm_15' -> 'slpm_15_f_1' etc.
            # The keys in latest_stats are generic 'slpm_15'.
            
            props_cols = [
                'slpm_15', 'sapm_15', 'td_avg_15', 'sub_avg_15',
                'f_1_chin_score', 'f_2_chin_score' # These are stored as 'chin_score'
            ]
            
            # Map generic to f1/f2
            for k in ['slpm_15', 'sapm_15', 'td_avg_15', 'sub_avg_15']:
                row[f'{k}_f_1'] = s1.get(k, 0)
                row[f'{k}_f_2'] = s2.get(k, 0)
                
            row['f_1_chin_score'] = s1.get('chin_score', 5)
            row['f_2_chin_score'] = s2.get('chin_score', 5)
            
            # Add odds
            row['f_1_odds'] = fight.f1_odds
            row['f_2_odds'] = fight.f2_odds
            
            # Add names
            row['f_1_name'] = f1
            row['f_2_name'] = f2
            
            # Create DF
            X_props_raw = pd.DataFrame([row])
            
            # Now call prepare_production_data
            # It calculates _adj and diffs.
            X_props, _ = prepare_production_data(X_props_raw)
            
            # Predict Props
            pp_win = prop_model_win.predict_proba(X_props)[:, 1][0]
            pp_finish = prop_model_finish.predict_proba(X_props)[:, 1][0]
            pp_method = prop_model_method.predict_proba(X_props)[0]
            pp_round = prop_model_round.predict_proba(X_props)[0]
            
            # ... (Same logic as before) ...
            if pp_win > 0.5:
                conf_w = pp_win
            else:
                conf_w = 1 - pp_win
                
            prob_ko = pp_finish * pp_method[0]
            prob_sub = pp_finish * pp_method[1]
            prob_dec = 1 - pp_finish
            
            methods = {'KO/TKO': prob_ko, 'Submission': prob_sub, 'Decision': prob_dec}
            best_method = max(methods, key=methods.get)
            conf_m = methods[best_method]
            
            import numpy as np
            best_rnd_idx = np.argmax(pp_round)
            best_rnd = best_rnd_idx + 1
            conf_r = pp_round[best_rnd_idx]
            
            pred_method = best_method
            
            if best_method == 'Decision':
                trifecta_prob = conf_w * prob_dec
                pred_round = "-"
            else:
                trifecta_prob = conf_w * conf_m * conf_r
                pred_round = str(best_rnd)
                
            # Min Odds
            if trifecta_prob > 0:
                min_odds_dec = 1 / trifecta_prob
                if min_odds_dec >= 2.0:
                    min_odds = f"+{int((min_odds_dec - 1) * 100)}"
                else:
                    min_odds = f"-{int(100 / (min_odds_dec - 1))}"
                    
        except Exception as e:
            print(f"Props Error: {e}")
            # Don't fail the whole request
            pass

    # Betting Logic
    is_value = False
    bet_target = ""
    edge = 0.0
    if p_f1 > 0.5:
        implied = 1 / fight.f1_odds
        if p_f1 > implied * 1.05:
            is_value = True
            bet_target = f1
            edge = p_f1 - implied
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
