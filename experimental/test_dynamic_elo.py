import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import json
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from experimental.features.dynamic_elo import calculate_dynamic_elo

def run_test():
    print("Loading data...")
    # Use raw data if possible to avoid pre-baked Elo, but we can just overwrite it.
    # UFC_data_with_elo.csv has 'elo_f1', 'elo_f2'. We will generate our own.
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 1. Calculate Dynamic Elo
    print("Calculating Dynamic Elo...")
    df_dynamic = calculate_dynamic_elo(df)
    
    # 2. Prepare Features
    # We want to compare:
    # Model A: Standard Elo (already in df as 'elo_f1', 'elo_f2', 'diff_elo')
    # Model B: Dynamic Elo ('dynamic_elo_f1', 'dynamic_elo_f2', 'diff_dynamic_elo')
    
    # We'll use a minimal feature set to isolate the impact of Elo.
    # Or we can use the full feature set and just swap the Elo columns.
    # Let's swap to see marginal improvement.
    
    with open('features_elo.json', 'r') as f:
        base_features = json.load(f)
        
    # Identify Elo features in base set
    elo_cols = [c for c in base_features if 'elo' in c]
    print(f"Base Elo Features: {elo_cols}")
    
    # Create Dynamic Feature Set
    # Replace 'elo_f1' with 'dynamic_elo_f1', etc.
    dynamic_features = []
    for feat in base_features:
        if 'elo' in feat:
            new_feat = feat.replace('elo', 'dynamic_elo')
            # Check if this column exists
            if new_feat in df_dynamic.columns:
                dynamic_features.append(new_feat)
            else:
                # If it's something like 'diff_elo', we made 'diff_dynamic_elo'
                # If it's 'elo_f1', we made 'dynamic_elo_f1'
                pass
        else:
            dynamic_features.append(feat)
            
    # Ensure we have the new columns
    # calculate_dynamic_elo adds: dynamic_elo_f1, dynamic_elo_f2, diff_dynamic_elo
    # If base_features has other elo stuff (e.g. rolling avg), we might miss it.
    # But let's assume standard set.
    
    print(f"Dynamic Features: {len(dynamic_features)}")
    
    # Encode
    cat_cols = df_dynamic[base_features].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_dynamic[col] = df_dynamic[col].astype(str)
        df_dynamic[col] = le.fit_transform(df_dynamic[col])
        
    # Target
    f1_wins = df_dynamic['winner'] == df_dynamic['f_1_name']
    f2_wins = df_dynamic['winner'] == df_dynamic['f_2_name']
    df_dynamic = df_dynamic[f1_wins | f2_wins].copy()
    df_dynamic['target'] = (df_dynamic['winner'] == df_dynamic['f_1_name']).astype(int)
    
    # Split
    split_idx = int(len(df_dynamic) * 0.85)
    train_df = df_dynamic.iloc[:split_idx]
    test_df = df_dynamic.iloc[split_idx:]
    
    y_train = train_df['target']
    y_test = test_df['target']
    
    # Model A: Standard Elo
    print("\nTraining Standard Elo Model...")
    xgb_std = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_std.fit(train_df[base_features], y_train)
    p_std = xgb_std.predict_proba(test_df[base_features])[:, 1]
    acc_std = accuracy_score(y_test, (p_std > 0.5).astype(int))
    ll_std = log_loss(y_test, p_std)
    print(f"Standard: Acc {acc_std:.4f}, LL {ll_std:.4f}")
    
    # Model B: Dynamic Elo
    print("\nTraining Dynamic Elo Model...")
    xgb_dyn = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_dyn.fit(train_df[dynamic_features], y_train)
    p_dyn = xgb_dyn.predict_proba(test_df[dynamic_features])[:, 1]
    acc_dyn = accuracy_score(y_test, (p_dyn > 0.5).astype(int))
    ll_dyn = log_loss(y_test, p_dyn)
    print(f"Dynamic: Acc {acc_dyn:.4f}, LL {ll_dyn:.4f}")
    
    # Save results
    with open('experimental/DYNAMIC_ELO_RESULTS.md', 'w') as f:
        f.write(f"# Dynamic Elo K-Factor Results\n\n")
        f.write(f"## Metrics\n")
        f.write(f"| Model | Accuracy | Log Loss |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| Standard Elo (Fixed K) | {acc_std:.4f} | {ll_std:.4f} |\n")
        f.write(f"| Dynamic Elo (Var K) | {acc_dyn:.4f} | {ll_dyn:.4f} |\n\n")
        
        f.write(f"## Interpretation\n")
        if ll_dyn < ll_std:
            f.write(f"Dynamic Elo improved Log Loss by {ll_std - ll_dyn:.4f}.\n")
        else:
            f.write(f"Dynamic Elo did not improve Log Loss (Diff: {ll_std - ll_dyn:.4f}).\n")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
