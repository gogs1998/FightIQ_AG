import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

def run_chin_health_experiment():
    print("=== Experiment: Chin Health Decay Model ===")
    
    # 1. Load Data
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # 2. Calculate Chin Health Features
    print("Calculating Chin Health Scores...")
    
    # We need to track cumulative damage for each fighter
    fighter_damage = {} # {fighter_name: {'ko_losses': 0, 'knockdowns_absorbed': 0}}
    
    chin_scores_f1 = []
    chin_scores_f2 = []
    
    # Iterate chronologically
    for idx, row in df.iterrows():
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        
        # Initialize if new
        if f1 not in fighter_damage: fighter_damage[f1] = {'ko_losses': 0, 'kd_absorbed': 0}
        if f2 not in fighter_damage: fighter_damage[f2] = {'ko_losses': 0, 'kd_absorbed': 0}
        
        # Get current stats (before this fight)
        d1 = fighter_damage[f1]
        d2 = fighter_damage[f2]
        
        # Formula: Start at 1.0. Decay by 10% for each KO loss, 2% for each KD absorbed.
        # Score = 1.0 * (0.9 ^ ko_losses) * (0.98 ^ kd_absorbed)
        score1 = 1.0 * (0.9 ** d1['ko_losses']) * (0.98 ** d1['kd_absorbed'])
        score2 = 1.0 * (0.9 ** d2['ko_losses']) * (0.98 ** d2['kd_absorbed'])
        
        chin_scores_f1.append(score1)
        chin_scores_f2.append(score2)
        
        # Update stats AFTER the fight
        res = str(row['result']).lower()
        
        # Did F1 get KO'd? (F2 won by KO)
        if 'ko' in res or 'tko' in res:
            if row['target'] == 0: # F2 won
                fighter_damage[f1]['ko_losses'] += 1
            else: # F1 won
                fighter_damage[f2]['ko_losses'] += 1
                
        # Knockdowns (We have f_1_kd and f_2_kd columns? Let's check)
        # Assuming f_1_kd is KDs SCORED by F1. So F2 absorbed them.
        kd1 = row.get('f_1_kd', 0)
        kd2 = row.get('f_2_kd', 0)
        
        if pd.notna(kd1): fighter_damage[f2]['kd_absorbed'] += kd1
        if pd.notna(kd2): fighter_damage[f1]['kd_absorbed'] += kd2
        
    df['f_1_chin_score'] = chin_scores_f1
    df['f_2_chin_score'] = chin_scores_f2
    df['diff_chin_score'] = df['f_1_chin_score'] - df['f_2_chin_score']
    
    print("Chin Scores Calculated.")
    
    # 3. Train Finish Model with Chin Features
    # We want to predict: Will the fight end by KO?
    # Target: 1 if KO/TKO, 0 otherwise (Decision or Sub)
    # Actually, let's target "Is KO?" specifically.
    
    def is_ko(res):
        r = str(res).lower()
        return 1 if ('ko' in r or 'tko' in r) else 0
        
    df['target_ko'] = df['result'].apply(is_ko)
    
    # Load Base Features
    with open(f'{BASE_DIR}/experiment_2/boruta_results.json', 'r') as f:
        base_features = json.load(f)['confirmed']
        
    new_features = ['f_1_chin_score', 'f_2_chin_score', 'diff_chin_score']
    all_features = base_features + new_features
    
    X = df[all_features].fillna(0)
    y = df['target_ko'].values
    
    # Split
    mask_train = df['event_date'] < '2024-01-01'
    mask_test = df['event_date'] >= '2024-01-01'
    
    print("\nTraining KO Prediction Model...")
    model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
    model.fit(X[mask_train], y[mask_train])
    
    # Evaluate
    preds = model.predict(X[mask_test])
    probs = model.predict_proba(X[mask_test])[:, 1]
    
    acc = accuracy_score(y[mask_test], preds)
    ll = log_loss(y[mask_test], probs)
    
    print(f"\n=== Chin Health Experiment Results (2024-2025) ===")
    print(f"Target: Predict KO/TKO (Binary)")
    print(f"Accuracy: {acc:.4%}")
    print(f"Log Loss: {ll:.4f}")
    
    # Feature Importance
    imp = pd.DataFrame({
        'Feature': all_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features for KO Prediction:")
    print(imp.head(10))
    
    # Check Chin Rank
    chin_rank = imp[imp['Feature'].isin(new_features)]
    print("\nChin Feature Ranks:")
    print(chin_rank)

if __name__ == "__main__":
    run_chin_health_experiment()
