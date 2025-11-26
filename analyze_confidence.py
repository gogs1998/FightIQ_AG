import pandas as pd
import joblib
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

def analyze_confidence():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
    model = joblib.load('ufc_model_elo.pkl')
    
    cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy()
    
    X_test = test_df[features]
    y_test = test_df['target']
    
    probs = model.predict_proba(X_test)[:, 1]
    test_df['prob_f1'] = probs
    test_df['pred_winner'] = np.where(probs > 0.5, 1, 0)
    test_df['correct'] = test_df['pred_winner'] == test_df['target']
    test_df['confidence'] = np.where(test_df['pred_winner'] == 1, test_df['prob_f1'], 1 - test_df['prob_f1'])
    
    # Bin confidence
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    test_df['conf_bin'] = pd.cut(test_df['confidence'], bins=bins, labels=labels)
    
    print("\n--- Accuracy by Confidence Level ---")
    stats = test_df.groupby('conf_bin')['correct'].agg(['count', 'mean'])
    stats['mean'] = stats['mean'].mul(100).round(2)
    stats.columns = ['Total Fights', 'Accuracy (%)']
    print(stats)

if __name__ == "__main__":
    analyze_confidence()
