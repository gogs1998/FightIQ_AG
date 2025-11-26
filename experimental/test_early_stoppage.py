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

from experimental.features.early_stoppage import calculate_stoppage_features

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 1. Calculate Stoppage Features
    print("Calculating Stoppage Features...")
    df_stop = calculate_stoppage_features(df)
    
    # 2. Define Feature Sets
    with open('features_elo.json', 'r') as f:
        base_features = json.load(f)
        
    stop_cols = [
        'f_1_finish_rate', 'f_1_been_finished_rate', 'f_1_avg_time',
        'f_2_finish_rate', 'f_2_been_finished_rate', 'f_2_avg_time',
        'diff_finish_rate', 'diff_been_finished_rate', 'diff_avg_time'
    ]
    
    # Create two feature lists
    # List A: Base Features
    features_A = base_features
    
    # List B: Base + Stoppage Features
    features_B = base_features + stop_cols
    
    print(f"Model A Features: {len(features_A)}")
    print(f"Model B Features: {len(features_B)}")
    
    # Encode
    cat_cols = df_stop[features_A].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_stop[col] = df_stop[col].astype(str)
        df_stop[col] = le.fit_transform(df_stop[col])
        
    # Target
    f1_wins = df_stop['winner'] == df_stop['f_1_name']
    f2_wins = df_stop['winner'] == df_stop['f_2_name']
    df_stop = df_stop[f1_wins | f2_wins].copy()
    df_stop['target'] = (df_stop['winner'] == df_stop['f_1_name']).astype(int)
    
    # Split
    split_idx = int(len(df_stop) * 0.85)
    train_df = df_stop.iloc[:split_idx]
    test_df = df_stop.iloc[split_idx:]
    
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
    
    # Model B: Stoppage Features
    print("\nTraining Stoppage Features Model...")
    xgb_stop = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_stop.fit(train_df[features_B], y_train)
    p_stop = xgb_stop.predict_proba(test_df[features_B])[:, 1]
    acc_stop = accuracy_score(y_test, (p_stop > 0.5).astype(int))
    ll_stop = log_loss(y_test, p_stop)
    print(f"Stoppage: Acc {acc_stop:.4f}, LL {ll_stop:.4f}")
    
    # Save results
    with open('experimental/STOPPAGE_RESULTS.md', 'w') as f:
        f.write(f"# Early Stoppage Propensity Results\n\n")
        f.write(f"## Metrics\n")
        f.write(f"| Model | Accuracy | Log Loss |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| Baseline | {acc_base:.4f} | {ll_base:.4f} |\n")
        f.write(f"| Stoppage Features | {acc_stop:.4f} | {ll_stop:.4f} |\n\n")
        
        f.write(f"## Interpretation\n")
        if ll_stop < ll_base:
            f.write(f"Stoppage Features improved Log Loss by {ll_base - ll_stop:.4f}.\n")
        else:
            f.write(f"Stoppage Features did not improve Log Loss (Diff: {ll_base - ll_stop:.4f}).\n")
            
        # Feature Importance
        imp = pd.Series(xgb_stop.feature_importances_, index=features_B).sort_values(ascending=False)
        f.write(f"\n## Stoppage Feature Importance\n")
        stop_imp = imp[imp.index.isin(stop_cols)]
        for name, val in stop_imp.sort_values(ascending=False).items():
            f.write(f"- {name}: {val:.4f}\n")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
