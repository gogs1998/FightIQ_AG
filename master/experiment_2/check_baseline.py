import pandas as pd
import numpy as np

def check_baseline():
    print("=== Baseline Check: The 'Chalk' Strategy ===")
    
    # 1. Load Data
    try:
        df = pd.read_csv('master/data/training_data.csv')
    except:
        df = pd.read_csv('data/training_data.csv')
        
    # Filter 2024-2025
    df['event_date'] = pd.to_datetime(df['event_date'])
    mask = (df['event_date'] >= '2024-01-01') & \
           (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
           (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
           
    test_df = df[mask].copy()
    
    print(f"Analyzing {len(test_df)} fights from 2024-2025...")
    
    correct = 0
    total = 0
    
    for idx, row in test_df.iterrows():
        # Who is the favorite?
        if row['f_1_odds'] < row['f_2_odds']:
            pred = 1 # Predict F1
        else:
            pred = 0 # Predict F2
            
        if pred == row['target']:
            correct += 1
        total += 1
        
    acc = correct / total
    print(f"\nBaseline Accuracy (Picking Favorite): {acc:.4%}")
    print(f"FightIQ Boruta Accuracy:            70.3571%")
    print(f"Edge over Market:                   {0.703571 - acc:.4%}")

if __name__ == "__main__":
    check_baseline()
