import pandas as pd
import numpy as np
import json
import os
import sys

try:
    from boruta import BorutaPy
except ImportError:
    print("Boruta package not found. Please install it using: pip install boruta")
    sys.exit(1)

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def run_boruta():
    print("Loading data...")
    data_path = 'v2/data/training_data_v2.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Prepare X and y
    # Drop non-features and targets
    exclude_cols = ['target', 'winner', 'winner_encoded', 'f_1_name', 'f_2_name', 'event_date', 'fight_id', 'event_id', 'weight_class', 'method', 'round', 'time']
    
    # Select numeric columns only
    X_df = df.select_dtypes(include=[np.number])
    
    # Drop columns that are in exclude list or look like IDs
    cols_to_drop = [c for c in X_df.columns if c in exclude_cols or 'id' in c.lower() and c not in ['f_1_odds', 'f_2_odds']]
    
    if 'target' in X_df.columns:
        cols_to_drop.append('target')
        
    X_df = X_df.drop(columns=cols_to_drop, errors='ignore')
    y = df['target'].values
    
    # Handle NaNs
    X_df = X_df.fillna(0)
    
    # Pre-filter: Drop constant columns
    X_df = X_df.loc[:, (X_df != X_df.iloc[0]).any()]
    
    X = X_df.values
    feature_names = X_df.columns.tolist()
    
    print(f"Starting Boruta on {X.shape[0]} rows and {X.shape[1]} features...")
    print("This may take a few minutes...")
    
    # Initialize Estimator
    # Faster config
    estimator = XGBClassifier(
        n_jobs=-1, 
        max_depth=3, # Reduced from 5
        n_estimators=50, # Reduced from 100
        eval_metric='logloss',
        random_state=42
    )
    
    # Initialize Boruta
    feat_selector = BorutaPy(
        estimator,
        n_estimators='auto',
        verbose=2,
        random_state=42,
        max_iter=30 # Reduced from 50
    )
    
    feat_selector.fit(X, y)
    
    # Results
    confirmed = [feature_names[i] for i, x in enumerate(feat_selector.support_) if x]
    tentative = [feature_names[i] for i, x in enumerate(feat_selector.support_weak_) if x]
    rejected = [feature_names[i] for i, x in enumerate(feat_selector.support_) if not x and not feat_selector.support_weak_[i]]
    
    print("\n=== Boruta Results ===")
    print(f"Confirmed Features: {len(confirmed)}")
    print(f"Tentative Features: {len(tentative)}")
    print(f"Rejected Features: {len(rejected)}")
    
    # Save
    results = {
        "confirmed": confirmed,
        "tentative": tentative,
        "rejected": rejected
    }
    
    os.makedirs('experimental', exist_ok=True)
    with open('experimental/boruta_features.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Saved to experimental/boruta_features.json")
    
    # Print top confirmed features
    print("\nTop 20 Confirmed Features:")
    print(confirmed[:20])

if __name__ == "__main__":
    run_boruta()
