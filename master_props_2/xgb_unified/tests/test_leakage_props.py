import pandas as pd
import numpy as np
import xgboost as xgb
import json
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

def run_leakage_tests():
    print("=== Comprehensive Leakage & Monkey Tests (Unified Props) ===")
    
    # 1. Load Data
    print("Loading data...")
    data_path = '../../../master_3/data/training_data_enhanced.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    print(f"Winner Encoded Values:\n{df['winner_encoded'].value_counts(dropna=False)}")
    print(f"Is Finish Values:\n{df['is_finish'].value_counts(dropna=False)}")
    
    # Load Features
    feat_path = '../../../master_3/features_enhanced.json'
    with open(feat_path, 'r') as f:
        features = json.load(f)
            
    print(f"Testing with {len(features)} features.")
    
    # 2. Feature Name Scan
    print("\n--- Test 1: Feature Name Scan ---")
    forbidden_terms = ['winner', 'result', 'outcome', 'decision', 'finish_round', 'time_seconds', 'is_finish', 'method']
    # Note: 'avg_time' is okay.
    
    leaky_names = []
    for f in features:
        for term in forbidden_terms:
            if term in f.lower() and 'avg' not in f.lower() and 'cum' not in f.lower() and 'rate' not in f.lower() and 'streak' not in f.lower():
                # Context checks
                if term == 'winner' and f == 'winner': leaky_names.append(f)
                elif term == 'result' and f == 'result': leaky_names.append(f)
                elif term == 'is_finish' and f == 'is_finish': leaky_names.append(f)
                elif term == 'finish_round' and f == 'finish_round': leaky_names.append(f)
                
    if leaky_names:
        print(f"FAIL: Found potential leaky feature names: {leaky_names}")
    else:
        print("PASS: No obvious leaky feature names found.")
        
    # 3. Correlation Check (All Targets)
    print("\n--- Test 2: Target Correlation Check ---")
    X = df[features].fillna(0)
    
    # Define Targets
    targets = {
        'Winner': df['winner_encoded'],
        'Finish': df['is_finish'],
        # Method/Round are subsets, check correlation on subset? Or full?
        # Full is fine, they are just 0/1 or 0-4.
        # But for Method, we need to encode it first.
        # Let's just check Winner and Finish for now as they are the main leakage vectors.
    }
    
    for t_name, y in targets.items():
        print(f"Checking {t_name} correlations...")
        corrs = []
        for col in features:
            try:
                c = X[col].corr(y)
                if abs(c) > 0.95:
                    corrs.append((col, c))
            except:
                pass
                
        if corrs:
            print(f"FAIL: Found features highly correlated with {t_name} (>0.95): {corrs}")
        else:
            print(f"PASS: No features have >0.95 correlation with {t_name}.")
            
    # 4. Monkey Test: Random Labels (All Models)
    print("\n--- Test 3: Monkey Test (Random Labels) ---", flush=True)
    
    models_to_test = [
        ('Winner', df['winner_encoded'].replace(-1, 0), {}),
        ('Finish', df['is_finish'], {}),
        # Method (Subset)
        ('Method', df['result'].apply(lambda x: 1 if 'Submission' in str(x) else 0), {'subset': df['is_finish']==1}),
        # Round (Subset)
        # Replace -1 with 0 (Round 1) to avoid crash. Filtering will remove invalid ones if logic works, but this is safer.
        ('Round', (pd.to_numeric(df['finish_round'], errors='coerce').fillna(0).astype(int) - 1).clip(-1, 4).replace(-1, 0), {'subset': df['is_finish']==1, 'objective': 'multi:softprob', 'num_class': 5})
    ]
    
    for name, y_target, opts in models_to_test:
        print(f"\nTesting {name} Model...", flush=True)
        
        subset_mask = opts.get('subset', slice(None))
        
        # Apply subset mask to both X and y
        if isinstance(subset_mask, pd.Series):
            X_sub = X[subset_mask]
            y_sub = y_target[subset_mask]
        else:
            X_sub = X
            y_sub = y_target
        
        # Filter out negative values (invalid rounds/methods/winners)
        # Winner has -1 for draws. Round has -1 for missing.
        if True: # Apply to all models to be safe
            valid_mask = y_sub >= 0
            X_sub = X_sub[valid_mask]
            y_sub = y_sub[valid_mask]
            
        baseline_acc = max(y_sub.value_counts(normalize=True).max(), 0)
        print(f"Majority Class Baseline: {baseline_acc:.4f}", flush=True)
        
        # Shuffle Labels
        y_shuffled = np.random.permutation(y_sub)
        
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_shuffled, test_size=0.2, random_state=42)
        
        print(f"DEBUG: y_train unique: {np.unique(y_train)}", flush=True)
        
        params = {'n_estimators': 10, 'max_depth': 3, 'n_jobs': -1}
        if 'objective' in opts: params['objective'] = opts['objective']
        if 'num_class' in opts: params['num_class'] = opts['num_class']
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Random Labels Accuracy: {acc:.4f}")
        if abs(acc - baseline_acc) < 0.05:
            print(f"PASS: {name} Model performs near baseline.")
        else:
            print(f"FAIL: {name} Model deviates from baseline ({baseline_acc:.4f}). Check for overfitting or leakage.")

    # 5. Monkey Test: Random Features
    print("\n--- Test 4: Monkey Test (Random Features) ---")
    
    for name, y_target, opts in models_to_test:
        print(f"\nTesting {name} Model...")
        
        subset_mask = opts.get('subset', slice(None))
        y_sub = y_target
        
        # Filter out negative values (invalid rounds/methods)
        if name in ['Round', 'Method']:
            # We need to filter y_sub, but X_random is generated fresh.
            # So we generate X_random matching the filtered y_sub length.
            valid_mask = y_sub >= 0
            y_sub = y_sub[valid_mask]
            
        n_samples = len(y_sub)
        baseline_acc = max(y_sub.value_counts(normalize=True).max(), 0)
        
        # Random Features
        X_random = pd.DataFrame(np.random.randn(n_samples, len(features)), columns=features)
        
        X_train, X_test, y_train, y_test = train_test_split(X_random, y_sub, test_size=0.2, random_state=42)
        
        params = {'n_estimators': 50, 'max_depth': 3, 'n_jobs': -1}
        if 'objective' in opts: params['objective'] = opts['objective']
        if 'num_class' in opts: params['num_class'] = opts['num_class']
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Random Features Accuracy: {acc:.4f}")
        if abs(acc - baseline_acc) < 0.05:
            print(f"PASS: {name} Model performs near baseline.")
        else:
            print(f"FAIL: {name} Model deviates from baseline ({baseline_acc:.4f}). Target might be leaking?")

if __name__ == "__main__":
    run_leakage_tests()
