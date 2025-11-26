import pandas as pd
import numpy as np
import json
import os
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel

def run_fast_selection():
    print("Loading data...")
    data_path = 'v2/data/training_data_v2.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Prepare X and y
    exclude_cols = ['target', 'winner', 'winner_encoded', 'f_1_name', 'f_2_name', 'event_date', 'fight_id', 'event_id', 'weight_class', 'method', 'round', 'time']
    
    X_df = df.select_dtypes(include=[np.number])
    cols_to_drop = [c for c in X_df.columns if c in exclude_cols or 'id' in c.lower() and c not in ['f_1_odds', 'f_2_odds']]
    
    if 'target' in X_df.columns:
        cols_to_drop.append('target')
    
    # CRITICAL: Remove fight-specific round columns (LEAKAGE)
    # These are stats FROM the fight being predicted, not historical
    fight_specific_patterns = [
        'f_1_r1_', 'f_1_r2_', 'f_1_r3_', 'f_1_r4_', 'f_1_r5_',
        'f_2_r1_', 'f_2_r2_', 'f_2_r3_', 'f_2_r4_', 'f_2_r5_',
        'f_1_knockdowns', 'f_2_knockdowns',  # Overall fight knockdowns
        'f_1_total_strikes_', 'f_2_total_strikes_',  # Overall fight strikes
        'f_1_sig_strikes_', 'f_2_sig_strikes_',  # Overall fight sig strikes
        'f_1_takedown_', 'f_2_takedown_',  # Overall fight takedowns
        'f_1_submission_', 'f_2_submission_',  # Overall fight submissions
        'f_1_ctrl_time_', 'f_2_ctrl_time_',  # Overall fight control time
        'f_1_reversals', 'f_2_reversals',  # Overall fight reversals
        'fight_duration_', 'r2_duration', 'r3_duration', 'r4_duration', 'r5_duration'  # Fight duration
    ]
    
    for pattern in fight_specific_patterns:
        cols_to_drop.extend([c for c in X_df.columns if pattern in c])
    
    X_df = X_df.drop(columns=cols_to_drop, errors='ignore')
    y = df['target'].values
    
    X_df = X_df.fillna(0)
    # Pre-filter constant columns
    X_df = X_df.loc[:, (X_df != X_df.iloc[0]).any()]
    
    X = X_df.values
    feature_names = X_df.columns.tolist()
    
    print(f"Starting Fast Selection on {X.shape[0]} rows and {X.shape[1]} features...")
    
    # Train XGBoost
    model = XGBClassifier(
        n_jobs=-1, 
        max_depth=5,
        n_estimators=100, 
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X, y)
    
    # Select Features
    # Select top features based on importance threshold
    # 'mean' selects features with importance > mean importance
    # We can also specify max_features
    selector = SelectFromModel(model, threshold='1.25*mean', prefit=True)
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i, x in enumerate(selected_mask) if x]
    
    print(f"\nSelected {len(selected_features)} features.")
    
    # Save
    results = {
        "confirmed": selected_features,
        "tentative": [],
        "rejected": [f for f in feature_names if f not in selected_features]
    }
    
    os.makedirs('experimental', exist_ok=True)
    with open('experimental/boruta_features.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Saved to experimental/boruta_features.json")
    print("\nTop 20 Selected Features:")
    # Sort by importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [feature_names[i] for i in indices[:20]]
    print(top_features)

if __name__ == "__main__":
    run_fast_selection()
