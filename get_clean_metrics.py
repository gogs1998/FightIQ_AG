import joblib
import pandas as pd
import json
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

def get_metrics():
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
    test_df = df.iloc[split_idx:]
    
    X_test = test_df[features]
    y_test = test_df['target']
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    print(f"ACCURACY: {acc:.5f}")
    print(f"LOG_LOSS: {ll:.5f}")

if __name__ == "__main__":
    get_metrics()
