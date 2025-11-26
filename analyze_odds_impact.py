import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def analyze_odds_impact():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Create Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    # Load features & model
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
    model = joblib.load('ufc_model_elo.pkl')
    
    # Handle Categoricals
    cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    # Split (Test set only)
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy()
    
    # Separate into With Odds and Without Odds
    has_odds = test_df[~test_df['f_1_odds'].isnull()].copy()
    no_odds = test_df[test_df['f_1_odds'].isnull()].copy()
    
    print(f"\nTotal Test Fights: {len(test_df)}")
    print(f"Fights WITH Odds: {len(has_odds)}")
    print(f"Fights WITHOUT Odds: {len(no_odds)}")
    
    # Evaluate With Odds
    if len(has_odds) > 0:
        X_odds = has_odds[features]
        y_odds = has_odds['target']
        preds_odds = model.predict(X_odds)
        acc_odds = accuracy_score(y_odds, preds_odds)
        print(f"\nAccuracy (WITH Odds): {acc_odds:.2%}")
        
    # Evaluate Without Odds
    if len(no_odds) > 0:
        X_no_odds = no_odds[features]
        y_no_odds = no_odds['target']
        preds_no_odds = model.predict(X_no_odds)
        acc_no_odds = accuracy_score(y_no_odds, preds_no_odds)
        print(f"Accuracy (WITHOUT Odds): {acc_no_odds:.2%}")
        
    # Baseline (Majority Class)
    majority_class = test_df['target'].mode()[0]
    baseline_acc = accuracy_score(test_df['target'], [majority_class]*len(test_df))
    print(f"\nBaseline Accuracy (Always pick {'F1' if majority_class==1 else 'F2'}): {baseline_acc:.2%}")

if __name__ == "__main__":
    analyze_odds_impact()
