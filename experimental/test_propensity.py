import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import json
import sys
import os

# Add current directory to path to import feature module
sys.path.append(os.getcwd())

from experimental.features.propensity import generate_fake_matches, fit_propensity_model, calculate_weights

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 1. Generate Propensity Model
    print("Generating fake matches for propensity modeling...")
    df_all = generate_fake_matches(df)
    print(f"Generated {len(df_all)} total samples (Real + Fake).")
    
    print("Training Propensity Model...")
    prop_model = fit_propensity_model(df_all)
    
    print("Calculating weights for real data...")
    weights, probs = calculate_weights(df, prop_model)
    df['sample_weight'] = weights
    df['propensity_score'] = probs
    
    print(f"Mean Propensity: {probs.mean():.4f}")
    print(f"Mean Weight: {weights.mean():.4f}")
    
    # 2. Prepare Data for XGBoost
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    # Encode categoricals
    cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    # Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    # Split
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    w_train = train_df['sample_weight']
    
    X_test = test_df[features]
    y_test = test_df['target']
    
    # 3. Train Baseline (Unweighted)
    print("\nTraining Baseline XGBoost (Unweighted)...")
    xgb_base = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=5,
        colsample_bytree=0.6, subsample=0.8, random_state=42, n_jobs=-1,
        early_stopping_rounds=50
    )
    xgb_base.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    base_probs = xgb_base.predict_proba(X_test)[:, 1]
    base_acc = accuracy_score(y_test, (base_probs > 0.5).astype(int))
    base_ll = log_loss(y_test, base_probs)
    print(f"Baseline Accuracy: {base_acc:.4f}")
    print(f"Baseline Log Loss: {base_ll:.4f}")
    
    # 4. Train Experiment (Weighted)
    print("\nTraining Weighted XGBoost (Inverse Propensity)...")
    xgb_weighted = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=5,
        colsample_bytree=0.6, subsample=0.8, random_state=42, n_jobs=-1,
        early_stopping_rounds=50
    )
    # Pass sample_weight to fit
    xgb_weighted.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_test, y_test)], verbose=False)
    
    w_probs = xgb_weighted.predict_proba(X_test)[:, 1]
    w_acc = accuracy_score(y_test, (w_probs > 0.5).astype(int))
    w_ll = log_loss(y_test, w_probs)
    print(f"Weighted Accuracy: {w_acc:.4f}")
    print(f"Weighted Log Loss: {w_ll:.4f}")
    
    # Save results
    with open('experimental/PROPENSITY_RESULTS.md', 'w') as f:
        f.write(f"# Propensity Weighting Results\n\n")
        f.write(f"## Metrics\n")
        f.write(f"| Model | Accuracy | Log Loss |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| Baseline | {base_acc:.4f} | {base_ll:.4f} |\n")
        f.write(f"| Weighted | {w_acc:.4f} | {w_ll:.4f} |\n\n")
        f.write(f"## Analysis\n")
        f.write(f"Mean Propensity Score: {probs.mean():.4f}\n")
        f.write(f"Mean Sample Weight: {weights.mean():.4f}\n")
        
        diff_acc = w_acc - base_acc
        diff_ll = w_ll - base_ll # Lower is better, so negative diff is good? No, usually we compare new - old.
        # If w_ll < base_ll, diff is negative (Good).
        
        f.write(f"\nImpact: Accuracy {'Improved' if diff_acc > 0 else 'Degraded'} by {diff_acc*100:.2f}%\n")
        f.write(f"Impact: Log Loss {'Improved' if diff_ll < 0 else 'Degraded'} by {diff_ll:.4f}\n")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
