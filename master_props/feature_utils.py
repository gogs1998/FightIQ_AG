import json
import pandas as pd
import sys
import os

# Add parent directory to path to import master_3 modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_production_feature_list():
    """
    Returns the exact list of features used by the production models.
    Combines base features, adjusted features, and prop-specific features.
    """
    # 1. Load Base Features
    # Try to find the json file relative to this script
    base_path = os.path.join(os.path.dirname(__file__), '..', 'master_3', 'features_selected.json')
    if not os.path.exists(base_path):
        # Fallback for running from different cwd
        base_path = 'master_3/features_selected.json'
        
    with open(base_path, 'r') as f:
        features = json.load(f)
        
    # 2. Add Adjusted Features (Opponent Adjusted)
    # These are the columns created by apply_opponent_adjustment
    adj_features = [
        'slpm_15_f_1_adj', 'slpm_15_f_2_adj', 
        'td_avg_15_f_1_adj', 'td_avg_15_f_2_adj', 
        'sub_avg_15_f_1_adj', 'sub_avg_15_f_2_adj', 
        'sapm_15_f_1_adj', 'sapm_15_f_2_adj'
    ]
    for f in adj_features:
        if f not in features:
            features.append(f)
            
    # 3. Add Prop Features (Chin Score, etc.)
    prop_features = ['f_1_chin_score', 'f_2_chin_score', 'diff_chin_score']
    for f in prop_features:
        if f not in features:
            features.append(f)
            
    # Ensure no duplicates and clean whitespace
    features = [f.strip() for f in features]
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in features:
        if f not in seen:
            unique_features.append(f)
            seen.add(f)
            
    return unique_features

def prepare_production_data(df):
    """
    Prepares a raw dataframe for prediction/training.
    1. Applies opponent adjustment.
    2. Ensures all required columns exist (filling with 0).
    3. Returns X (features only) and the modified df.
    """
    df = df.copy()
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Apply Manual Opponent Adjustment
    # (Copying logic from generate_full_predictions.py to ensure consistency)
    elo_col = 'dynamic_elo'
    # Use global mean if available, else local mean
    if f'{elo_col}_f1' in df.columns:
        avg_elo = df[f'{elo_col}_f1'].mean()
    else:
        # Fallback default if running on single row without context
        avg_elo = 1500.0 
        
    stat_cols = [
        'slpm_15_f_1', 'sapm_15_f_1', 'td_avg_15_f_1', 'sub_avg_15_f_1',
        'slpm_15_f_2', 'sapm_15_f_2', 'td_avg_15_f_2', 'sub_avg_15_f_2'
    ]
    
    for col in stat_cols:
        if col not in df.columns:
            df[col] = 0.0 # Safety fill
            
        if col.startswith('f_1_') or col.endswith('_f_1'):
            opp_elo = df.get(f'{elo_col}_f2', 1500.0)
            df[f'{col}_adj'] = df[col] * (opp_elo / avg_elo)
        elif col.startswith('f_2_') or col.endswith('_f_2'):
            opp_elo = df.get(f'{elo_col}_f1', 1500.0)
            df[f'{col}_adj'] = df[col] * (opp_elo / avg_elo)
            
    # Get Feature List
    feature_list = get_production_feature_list()
    
    # Create X dataframe
    X = pd.DataFrame(index=df.index)
    for col in feature_list:
        if col in df.columns:
            X[col] = df[col]
        else:
            # print(f"Warning: Feature {col} missing, filling with 0")
            X[col] = 0.0
            
    # Ensure numeric
    X = X.astype(float)
    
    return X, df
