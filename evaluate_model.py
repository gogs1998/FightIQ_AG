import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import LabelEncoder

def load_and_prep_data(path):
    print("Loading data...")
    df = pd.read_csv(path)
    
    # Convert date
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Create Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    return df

def evaluate():
    print("Loading model...")
    model = joblib.load('ufc_prediction_model.pkl')
    feature_names = model.get_booster().feature_names
    
    print("Loading data...")
    df = load_and_prep_data('UFC_full_data_golden.csv')
    
    # Prepare features
    X = df[feature_names].copy()
    y = df['target']
    
    # Handle categoricals
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        
    # Split (same as training)
    split_idx = int(len(df) * 0.85)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"Test set size: {len(X_test)}")
    
    # Predict
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    evaluate()
