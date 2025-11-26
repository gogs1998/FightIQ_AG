import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import json
import sys
import os
from sklearn.impute import SimpleImputer

# Add current directory to path
sys.path.append(os.getcwd())

from experimental.models.stacking import generate_oof_preds, fit_stack

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
    
    # Split
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    dates_train = train_df['event_date']
    
    X_test = test_df[features]
    y_test = test_df['target']
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    


    # Define Base Models
    models = {
        'xgb_deep': xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.01, max_depth=7,
            colsample_bytree=0.6, subsample=0.8, random_state=42, n_jobs=-1
        ),
        'xgb_shallow': xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=3,
            colsample_bytree=0.8, subsample=0.8, random_state=42, n_jobs=-1
        ),
        'linear': make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LogisticRegression(max_iter=1000, C=0.1))
    }
    
    # 1. Generate OOF Preds for Training Set
    print("\n--- Generating OOF Predictions (Time-Anchored) ---")
    oof_train = generate_oof_preds(models, X_train, y_train, dates_train, n_splits=5)
    
    # 2. Train Base Models on Full Training Set
    print("\n--- Retraining Base Models on Full Train Set ---")
    trained_models = {}
    test_preds = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict on Holdout
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_test)[:, 1]
        else:
            p = model.predict(X_test)
        test_preds[name] = p
        
        acc = accuracy_score(y_test, (p > 0.5).astype(int))
        ll = log_loss(y_test, p)
        print(f"  {name} Holdout - Acc: {acc:.4f}, LL: {ll:.4f}")

    test_meta = pd.DataFrame(test_preds)
    
    # 3. Train Meta-Learner
    print("\n--- Training Meta-Learner ---")
    meta_model = fit_stack(oof_train, y_train)
    
    print(f"Meta-Learner Coefficients: {meta_model.coef_}")
    print(f"Meta-Learner Intercept: {meta_model.intercept_}")
    
    # 4. Predict on Holdout using Meta-Learner
    stack_probs = meta_model.predict_proba(test_meta)[:, 1]
    stack_acc = accuracy_score(y_test, (stack_probs > 0.5).astype(int))
    stack_ll = log_loss(y_test, stack_probs)
    
    print(f"\n--- Stacking Results ---")
    print(f"Stacking Accuracy: {stack_acc:.4f}")
    print(f"Stacking Log Loss: {stack_ll:.4f}")
    
    # Find best single model
    best_single_acc = 0
    best_single_ll = 100
    best_name = ""
    
    for name, p in test_preds.items():
        acc = accuracy_score(y_test, (p > 0.5).astype(int))
        ll = log_loss(y_test, p)
        if ll < best_single_ll:
            best_single_ll = ll
            best_single_acc = acc
            best_name = name
            
    print(f"Best Single Model ({best_name}): Acc {best_single_acc:.4f}, LL {best_single_ll:.4f}")
    
    # Save results
    with open('experimental/STACKING_RESULTS.md', 'w') as f:
        f.write(f"# Time-Anchored Stacking Results\n\n")
        f.write(f"## Metrics\n")
        f.write(f"| Model | Accuracy | Log Loss |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        for name, p in test_preds.items():
            acc = accuracy_score(y_test, (p > 0.5).astype(int))
            ll = log_loss(y_test, p)
            f.write(f"| {name} | {acc:.4f} | {ll:.4f} |\n")
        f.write(f"| **Stacking** | **{stack_acc:.4f}** | **{stack_ll:.4f}** |\n\n")
        
        f.write(f"## Meta-Learner Weights\n")
        for name, coef in zip(models.keys(), meta_model.coef_[0]):
            f.write(f"- {name}: {coef:.4f}\n")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
