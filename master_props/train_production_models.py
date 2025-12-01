import pandas as pd
import xgboost as xgb
import joblib
import os
import sys
from feature_utils import prepare_production_data

def train_production_models():
    print("=== Master Props: Training Production Models ===")
    
    # 1. Load Data
    try:
        df = pd.read_csv('../master_3/data/training_data_enhanced.csv')
    except:
        df = pd.read_csv('data/training_data_enhanced.csv')
        
    # 2. Prepare Data
    print("Preparing data...")
    # Filter for training period (2010-2023)
    df['event_date'] = pd.to_datetime(df['event_date'])
    train_mask = df['event_date'] < '2024-01-01'
    train_df_raw = df[train_mask].copy()
    
    X_train_full, train_df = prepare_production_data(train_df_raw)
    
    print(f"Training Data Shape: {X_train_full.shape}")
    
    # 3. Define Targets
    def get_winner_target(row):
        if row['winner'] == row['f_1_name']: return 1
        if row['winner'] == row['f_2_name']: return 0
        return -1 
        
    def get_finish_target(row):
        res = str(row['result']).lower()
        if 'decision' in res: return 0
        if 'draw' in res or 'no contest' in res: return -1
        return 1
        
    def get_method_target(row):
        res = str(row['result']).lower()
        if 'ko' in res or 'tko' in res: return 0 # KO
        if 'submission' in res: return 1 # Sub
        return -1 
        
    def get_round_target(row):
        try:
            r = int(row['finish_round'])
            if 1 <= r <= 5: return r - 1 # 0-indexed class
            return -1
        except:
            return -1

    train_df['win_target'] = train_df.apply(get_winner_target, axis=1)
    train_df['finish_target'] = train_df.apply(get_finish_target, axis=1)
    train_df['method_target'] = train_df.apply(get_method_target, axis=1)
    train_df['round_target'] = train_df.apply(get_round_target, axis=1)
    
    # 4. Train Models
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Winner Model (Boosted)
    print("Training Winner Model...")
    mask_win = train_df['win_target'] != -1
    model_win = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model_win.fit(X_train_full[mask_win], train_df.loc[mask_win, 'win_target'])
    joblib.dump(model_win, f'{models_dir}/production_winner.pkl')
    
    # Finish Model
    print("Training Finish Model...")
    mask_fin = train_df['finish_target'] != -1
    model_finish = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    model_finish.fit(X_train_full[mask_fin], train_df.loc[mask_fin, 'finish_target'])
    joblib.dump(model_finish, f'{models_dir}/production_finish.pkl')
    
    # Method Model
    print("Training Method Model...")
    mask_meth = train_df['method_target'] != -1
    model_method = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    model_method.fit(X_train_full[mask_meth], train_df.loc[mask_meth, 'method_target'])
    joblib.dump(model_method, f'{models_dir}/production_method.pkl')
    
    # Round Model
    print("Training Round Model...")
    mask_rnd = train_df['round_target'] != -1
    model_round = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, objective='multi:softprob', num_class=5)
    model_round.fit(X_train_full[mask_rnd], train_df.loc[mask_rnd, 'round_target'])
    joblib.dump(model_round, f'{models_dir}/production_round.pkl')
    
    print("All models trained and saved to master_props/models/")

if __name__ == "__main__":
    train_production_models()
