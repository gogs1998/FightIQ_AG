import optuna
import pandas as pd
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit

def optimize_boruta():
    print("=== FightIQ Boruta Model Optimization (Optuna) ===")
    
    # 1. Load Data & Features
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    with open(f'{BASE_DIR}/experiment_2/boruta_results.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    print(f"Optimizing model with {len(features)} features...")
    
    # Filter valid odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    # 2. Define Objective
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'n_jobs': -1,
            'random_state': 42
        }
        
        # Time Series CV
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            scores.append(acc)
            
        return np.mean(scores)

    # 3. Run Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30) # 30 trials for speed
    
    print("\n=== Best Parameters ===")
    print(study.best_params)
    print(f"Best CV Accuracy: {study.best_value:.4%}")
    
    # 4. Save Params
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    print("Saved to experiment_2/boruta_params.json")

if __name__ == "__main__":
    optimize_boruta()
