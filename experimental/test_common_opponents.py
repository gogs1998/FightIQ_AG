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

from experimental.features.common_opponents import calculate_common_opponent_features

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 1. Calculate Common Opponent Features
    print("Calculating Common Opponent Features...")
    df_common = calculate_common_opponent_features(df)
    
    # 2. Define Feature Sets
    with open('features_elo.json', 'r') as f:
        base_features = json.load(f)
        
    common_cols = ['triangle_score', 'n_common_opponents', 'common_win_pct_diff']
    
    # Create two feature lists
    # List A: Base Features
    features_A = base_features
    
    # List B: Base + Common Opponents
    features_B = base_features + common_cols
    
    print(f"Model A Features: {len(features_A)}")
    print(f"Model B Features: {len(features_B)}")
    
    # Encode
    # We need to handle object columns in base_features
    cat_cols = df_common[features_A].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_common[col] = df_common[col].astype(str)
        df_common[col] = le.fit_transform(df_common[col])
        
    # Target
    f1_wins = df_common['winner'] == df_common['f_1_name']
    f2_wins = df_common['winner'] == df_common['f_2_name']
    df_common = df_common[f1_wins | f2_wins].copy()
    df_common['target'] = (df_common['winner'] == df_common['f_1_name']).astype(int)
    
    # Split
    split_idx = int(len(df_common) * 0.85)
    train_df = df_common.iloc[:split_idx]
    test_df = df_common.iloc[split_idx:]
    
    y_train = train_df['target']
    y_test = test_df['target']
    
    # Model A: Baseline
    print("\nTraining Baseline Model...")
    xgb_base = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_base.fit(train_df[features_A], y_train)
    p_base = xgb_base.predict_proba(test_df[features_A])[:, 1]
    acc_base = accuracy_score(y_test, (p_base > 0.5).astype(int))
    ll_base = log_loss(y_test, p_base)
    print(f"Baseline: Acc {acc_base:.4f}, LL {ll_base:.4f}")
    
    # Model B: Common Opponents
    print("\nTraining Common Opponents Model...")
    xgb_common = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_common.fit(train_df[features_B], y_train)
    p_common = xgb_common.predict_proba(test_df[features_B])[:, 1]
    acc_common = accuracy_score(y_test, (p_common > 0.5).astype(int))
    ll_common = log_loss(y_test, p_common)
    print(f"Common Opps: Acc {acc_common:.4f}, LL {ll_common:.4f}")
    
    # Save results
    with open('experimental/COMMON_OPP_RESULTS.md', 'w') as f:
        f.write(f"# Universal Opponent Math Results\n\n")
        f.write(f"## Metrics\n")
        f.write(f"| Model | Accuracy | Log Loss |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| Baseline | {acc_base:.4f} | {ll_base:.4f} |\n")
        f.write(f"| Common Opponents | {acc_common:.4f} | {ll_common:.4f} |\n\n")
        
        f.write(f"## Interpretation\n")
        if ll_common < ll_base:
            f.write(f"Common Opponents improved Log Loss by {ll_base - ll_common:.4f}.\n")
        else:
            f.write(f"Common Opponents did not improve Log Loss (Diff: {ll_base - ll_common:.4f}).\n")
            
        # Feature Importance
        imp = pd.Series(xgb_common.feature_importances_, index=features_B).sort_values(ascending=False)
        f.write(f"\n## Common Opponent Feature Importance\n")
        common_imp = imp[imp.index.isin(common_cols)]
        for name, val in common_imp.sort_values(ascending=False).items():
            f.write(f"- {name}: {val:.4f}\n")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
