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

from experimental.features.strength_adjustment import calculate_adjusted_stats

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 1. Calculate Adjusted Stats
    print("Calculating Strength-Adjusted Stats...")
    df_adj = calculate_adjusted_stats(df, elo_col_f1='f_1_elo', elo_col_f2='f_2_elo')
    
    # 2. Define Feature Sets
    with open('features_elo.json', 'r') as f:
        base_features = json.load(f)
        
    # Baseline Features (Standard Stats)
    # We want to isolate the stats part.
    # Let's identify the stats columns in base_features
    stats_cols = [
        'f_1_fighter_SlpM', 'f_1_fighter_TD_Avg', 
        'f_2_fighter_SlpM', 'f_2_fighter_TD_Avg',
        'diff_fighter_SlpM', 'diff_fighter_TD_Avg'
    ]
    
    # Check if they exist in base_features
    stats_cols = [c for c in stats_cols if c in base_features]
    print(f"Baseline Stats Features: {stats_cols}")
    
    # Adjusted Features
    # We want to replace the baseline stats with our adjusted ones.
    # Our new cols are: f_1_adj_slpm_avg, f_1_adj_td_avg, etc.
    # And diffs: diff_adj_slpm_avg, diff_adj_td_avg
    
    adj_cols = []
    for c in stats_cols:
        if 'SlpM' in c:
            new_c = c.replace('fighter_SlpM', 'adj_slpm_raw_avg')
        elif 'TD_Avg' in c:
            new_c = c.replace('fighter_TD_Avg', 'adj_td_raw_avg')
        else:
            continue
            
        if new_c in df_adj.columns:
            adj_cols.append(new_c)
            
    print(f"Adjusted Stats Features: {adj_cols}")
    
    # Create two feature lists
    # List A: Base Features (Standard)
    features_A = base_features
    
    # List B: Adjusted Features (Replace Standard with Adjusted)
    features_B = [c for c in base_features if c not in stats_cols] + adj_cols
    
    print(f"Model A Features: {len(features_A)}")
    print(f"Model B Features: {len(features_B)}")
    
    # Encode
    cat_cols = df_adj[features_A].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_adj[col] = df_adj[col].astype(str)
        df_adj[col] = le.fit_transform(df_adj[col])
        
    # Target
    f1_wins = df_adj['winner'] == df_adj['f_1_name']
    f2_wins = df_adj['winner'] == df_adj['f_2_name']
    df_adj = df_adj[f1_wins | f2_wins].copy()
    df_adj['target'] = (df_adj['winner'] == df_adj['f_1_name']).astype(int)
    
    # Split
    split_idx = int(len(df_adj) * 0.85)
    train_df = df_adj.iloc[:split_idx]
    test_df = df_adj.iloc[split_idx:]
    
    y_train = train_df['target']
    y_test = test_df['target']
    
    # Model A: Standard Stats
    print("\nTraining Standard Stats Model...")
    xgb_std = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_std.fit(train_df[features_A], y_train)
    p_std = xgb_std.predict_proba(test_df[features_A])[:, 1]
    acc_std = accuracy_score(y_test, (p_std > 0.5).astype(int))
    ll_std = log_loss(y_test, p_std)
    print(f"Standard: Acc {acc_std:.4f}, LL {ll_std:.4f}")
    
    # Model B: Adjusted Stats
    print("\nTraining Adjusted Stats Model...")
    xgb_adj = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_adj.fit(train_df[features_B], y_train)
    p_adj = xgb_adj.predict_proba(test_df[features_B])[:, 1]
    acc_adj = accuracy_score(y_test, (p_adj > 0.5).astype(int))
    ll_adj = log_loss(y_test, p_adj)
    print(f"Adjusted: Acc {acc_adj:.4f}, LL {ll_adj:.4f}")
    
    # Save results
    with open('experimental/STRENGTH_ADJ_RESULTS.md', 'w') as f:
        f.write(f"# Opponent Strength Adjustment Results\n\n")
        f.write(f"## Metrics\n")
        f.write(f"| Model | Accuracy | Log Loss |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| Standard Stats | {acc_std:.4f} | {ll_std:.4f} |\n")
        f.write(f"| Adjusted Stats | {acc_adj:.4f} | {ll_adj:.4f} |\n\n")
        
        f.write(f"## Interpretation\n")
        if ll_adj < ll_std:
            f.write(f"Adjusted Stats improved Log Loss by {ll_std - ll_adj:.4f}.\n")
        else:
            f.write(f"Adjusted Stats did not improve Log Loss (Diff: {ll_std - ll_adj:.4f}).\n")
            
        # Feature Importance
        imp = pd.Series(xgb_adj.feature_importances_, index=features_B).sort_values(ascending=False)
        f.write(f"\n## Top Adjusted Features\n")
        adj_imp = imp[imp.index.isin(adj_cols)]
        for name, val in adj_imp.sort_values(ascending=False).items():
            f.write(f"- {name}: {val:.4f}\n")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
