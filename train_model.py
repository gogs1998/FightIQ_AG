import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def load_and_prep_data(path):
    print("Loading data...")
    df = pd.read_csv(path)
    
    # Convert date
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Create Target
    # 1 if Fighter 1 wins, 0 if Fighter 2 wins
    # Drop rows where winner is neither (Draws, NC)
    print("Creating target...")
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    print(f"Data shape after filtering draws/NC: {df.shape}")
    print(f"Target distribution: {df['target'].mean():.4f} (1 = F1 wins)")
    
    return df

def select_features(df):
    print("Selecting features...")
    cols = list(df.columns)
    
    safe_keywords = ['avg', 'odds', 'age', 'height', 'reach', 'weight', 'title', 'gender', 'streak', 'wins', 'losses', 'draws', 'diff']
    unsafe_keywords = ['winner', 'result', 'finish', 'score', 'decision', 'bonus', 'target', 'f_1_name', 'f_2_name', 'referee', 'date', 'location', 'url']
    
    features = []
    for c in cols:
        c_lower = c.lower()
        
        # Skip target and metadata
        if c == 'target': continue
        
        # Must not contain obvious leakage keywords
        if any(k in c_lower for k in unsafe_keywords):
            continue
            
        # STRICT LEAKAGE CHECK
        # If it contains any stat-like keyword, it MUST also contain an aggregation keyword
        stat_keywords = [
            'landed', 'attempted', 'knockdowns', 'sub_att', 'rev', 'ctrl',
            'succ', 'att', 'acc', 'share', 'pct', 'strikes', 'ground', 'head', 'leg', 'distance', 'clinch'
        ]
        aggregation_keywords = ['avg', 'cum', 'per_min', 'diff'] # 'diff' might be risky if it's diff of current stats, but usually diff of avgs
        
        is_stat = any(k in c_lower for k in stat_keywords)
        is_aggregated = any(k in c_lower for k in aggregation_keywords)
        
        if is_stat and not is_aggregated:
            continue
            
        # Extra check for Round specific stats which are almost always leakage if not averaged
        # e.g. f_1_r1_...
        if '_r1_' in c_lower or '_r2_' in c_lower or '_r3_' in c_lower or '_r4_' in c_lower or '_r5_' in c_lower:
             if not is_aggregated:
                continue

        features.append(c)
        
    print(f"Selected {len(features)} features.")
    return features

def train_model():
    path = 'UFC_full_data_golden.csv'
    df = load_and_prep_data(path)
    
    features = select_features(df)
    
    # Identify categorical columns
    cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {len(cat_cols)}")
    
    # Label Encode Categoricals
    for col in cat_cols:
        le = LabelEncoder()
        # Handle unknown categories in test/future by converting to string
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    # Time-based Split
    # Use last 15% as test set
    split_idx = int(len(df) * 0.85)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train XGBoost
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    print("\n--- Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # Feature Importance
    print("\nTop 20 Features:")
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance.head(20))
    
    # Save model
    joblib.dump(model, 'ufc_prediction_model.pkl')
    print("\nModel saved to ufc_prediction_model.pkl")

if __name__ == "__main__":
    train_model()
