import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score

def verify():
    print("Loading data...")
    df = pd.read_csv('UFC_full_data_golden.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Create Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    # Split
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"\n--- Split Verification ---")
    print(f"Train Date Range: {train_df['event_date'].min().date()} to {train_df['event_date'].max().date()}")
    print(f"Test Date Range:  {test_df['event_date'].min().date()} to {test_df['event_date'].max().date()}")
    print(f"Train Size: {len(train_df)}")
    print(f"Test Size: {len(test_df)}")
    
    # Load Model and Evaluate
    print("\nLoading model...")
    model = joblib.load('ufc_model_optimized.pkl')
    with open('final_features.json', 'r') as f:
        features = json.load(f)
        
    X_test = test_df[features]
    y_test = test_df['target']
    
    # Handle categoricals if needed (simple string conversion for safety)
    cat_cols = X_test.select_dtypes(include=['object']).columns
    for c in cat_cols:
        X_test[c] = X_test[c].astype(str)
        # Note: LabelEncoder from training isn't saved here, which is a potential issue if we have categoricals.
        # But we filtered out most non-numeric features. Let's see if it runs.
        # If the model expects encoded integers, passing strings will fail.
        # However, XGBoost native support might handle it if configured, but we used sklearn API with LabelEncoder.
        # We should check if we have categoricals in final_features.
    
    try:
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\nHoldout Accuracy: {acc:.4%}")
    except Exception as e:
        print(f"Evaluation failed (likely due to encoding): {e}")

if __name__ == "__main__":
    verify()
