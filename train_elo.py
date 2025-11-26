import pandas as pd
import xgboost as xgb
import joblib
import json
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import LabelEncoder

def train_with_elo():
    print("Loading data with Elo...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Create Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    # Load previous top features
    with open('final_features.json', 'r') as f:
        features = json.load(f)
        
    # REMOVE LEAKAGE: Duration features
    # 'r1_duration', 'r2_duration', etc. are likely the duration of the current fight's rounds.
    # Even if they are averages, they are ambiguous and risky.
    # Given they were flagged in deep audit, we remove them to be safe.
    features = [f for f in features if 'duration' not in f]
        
    # Add Elo features
    new_features = ['f_1_elo', 'f_2_elo', 'diff_elo']
    features.extend(new_features)
    
    print(f"Training with {len(features)} features (including new Elo features).")
    
    # Handle Categoricals
    cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    # Split
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']
    
    # Train
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\nNew Accuracy with Elo: {acc:.4f}")
    
    # Check importance of Elo
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(importance.head(10))
    
    # Save
    joblib.dump(model, 'ufc_model_elo.pkl')
    with open('features_elo.json', 'w') as f:
        json.dump(features, f)

if __name__ == "__main__":
    train_with_elo()
