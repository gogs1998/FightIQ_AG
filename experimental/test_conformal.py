import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from experimental.decision.conformal import ConformalClassifier

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Load features
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
    
    # Split: Train (70%), Calibration (15%), Test (15%)
    # Time-based split
    n = len(df)
    idx_train = int(n * 0.70)
    idx_cal = int(n * 0.85)
    
    train_df = df.iloc[:idx_train]
    cal_df = df.iloc[idx_train:idx_cal]
    test_df = df.iloc[idx_cal:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    
    X_cal = cal_df[features]
    y_cal = cal_df['target']
    
    X_test = test_df[features]
    y_test = test_df['target']
    
    print(f"Split sizes: Train={len(X_train)}, Cal={len(X_cal)}, Test={len(X_test)}")
    
    # 1. Train Model
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3,
        colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 2. Get Probabilities
    # We need (n, 2) probabilities for conformal
    prob_cal = model.predict_proba(X_cal)
    prob_test = model.predict_proba(X_test)
    
    # 3. Calibrate Conformal Predictor
    print("Calibrating Conformal Predictor (Target Coverage: 90%)...")
    cc = ConformalClassifier(alpha=0.1)
    cc.fit(prob_cal, y_cal.values)
    
    # 4. Predict Sets
    sets_test = cc.predict(prob_test) # Boolean mask (n, 2)
    
    # 5. Evaluate
    # Coverage: Is true label in set?
    # y_test is 0 or 1.
    # sets_test[i, y_test[i]] should be True
    
    covered = sets_test[np.arange(len(y_test)), y_test.values]
    coverage = covered.mean()
    
    # Set Size
    set_sizes = sets_test.sum(axis=1)
    avg_size = set_sizes.mean()
    
    # Distribution
    size_counts = pd.Series(set_sizes).value_counts().sort_index()
    
    print(f"\n--- Conformal Results ---")
    print(f"Target Coverage: 90%")
    print(f"Actual Coverage: {coverage*100:.2f}%")
    print(f"Average Set Size: {avg_size:.4f}")
    print(f"Set Size Distribution:\n{size_counts}")
    
    # Save results
    with open('experimental/CONFORMAL_RESULTS.md', 'w') as f:
        f.write(f"# Split Conformal Prediction Results\n\n")
        f.write(f"## Metrics\n")
        f.write(f"- **Target Coverage**: 90%\n")
        f.write(f"- **Actual Coverage**: {coverage*100:.2f}%\n")
        f.write(f"- **Average Set Size**: {avg_size:.4f}\n\n")
        f.write(f"## Set Size Distribution\n")
        for size, count in size_counts.items():
            pct = count / len(y_test) * 100
            desc = "Empty" if size==0 else ("Singleton (Bet)" if size==1 else "Both (Abstain)")
            f.write(f"- Size {size} ({desc}): {count} ({pct:.2f}%)\n")
            
        f.write(f"\n## Interpretation\n")
        f.write(f"The model is confident enough to make a single prediction in {size_counts.get(1, 0) / len(y_test) * 100:.1f}% of cases.\n")
        f.write(f"It abstains (predicts both) in {size_counts.get(2, 0) / len(y_test) * 100:.1f}% of cases to maintain 90% coverage.\n")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
