import pandas as pd
import joblib
import json
import numpy as np

def analyze_errors():
    print("Loading data...")
    df = pd.read_csv('UFC_full_data_golden.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Create Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    # Split (Test set only)
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy()
    
    # Load Model
    model = joblib.load('ufc_model_optimized.pkl')
    with open('final_features.json', 'r') as f:
        features = json.load(f)
        
    X_test = test_df[features]
    
    # Predict
    probs = model.predict_proba(X_test)[:, 1]
    test_df['prob_f1'] = probs
    test_df['pred_winner'] = np.where(probs > 0.5, 1, 0)
    test_df['correct'] = test_df['pred_winner'] == test_df['target']
    test_df['confidence'] = np.where(test_df['pred_winner'] == 1, test_df['prob_f1'], 1 - test_df['prob_f1'])
    
    # Analyze High Confidence Errors
    errors = test_df[~test_df['correct']].copy()
    errors = errors.sort_values('confidence', ascending=False)
    
    print("\n--- Top 10 High Confidence Errors ---")
    cols_to_show = ['event_date', 'f_1_name', 'f_2_name', 'winner', 'confidence', 'f_1_odds', 'f_2_odds']
    print(errors[cols_to_show].head(10))
    
    # Analyze by Weight Class if available
    # We need to find the weight class column. It was likely encoded or dropped if not in top features.
    # Let's check original df
    if 'weight_class' in df.columns:
        print("\n--- Accuracy by Weight Class ---")
        test_df['weight_class'] = df.loc[test_df.index, 'weight_class']
        acc_by_class = test_df.groupby('weight_class')['correct'].mean().sort_values()
        print(acc_by_class)

if __name__ == "__main__":
    analyze_errors()
