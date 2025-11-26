import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

# Add current dir to path to import features
sys.path.append(os.path.join(os.getcwd(), 'experimental'))
from features.rest import fit_rest_priors

def calculate_rest_expanding(df):
    """
    Calculate REST stats for each fight using only PAST history.
    """
    df = df.sort_values('event_date')
    
    # We need 'finished' column (1 if not decision, 0 if decision)
    # We can infer this from 'result' or 'outcome_method'
    # 'result' usually contains 'Decision', 'KO/TKO', 'Submission'
    # Let's check what we have.
    
    # If 'finished' not in df, create it
    if 'finished' not in df.columns:
        # Look for outcome method
        if 'outcome_method' in df.columns:
            df['finished'] = df['outcome_method'].apply(lambda x: 0 if 'Decision' in str(x) else 1)
        elif 'result' in df.columns:
            df['finished'] = df['result'].apply(lambda x: 0 if 'Decision' in str(x) else 1)
        else:
            print("WARNING: No result/outcome column found to determine 'finished'. REST will fail.")
            return pd.DataFrame()
            
    rest_features = []
    
    # Expanding window
    # We want to calculate priors based on ALL past fights.
    # But re-calculating for every single fight is slow.
    # We can do it in batches (e.g. per month) or just use a simple expanding mean.
    
    # Actually, fit_rest_priors groups by referee/venue.
    # We can iterate through unique dates.
    
    print("Calculating REST features...")
    
    dates = df['event_date'].unique()
    
    # Pre-calculate groups to save time?
    # No, priors change over time.
    
    # Optimization:
    # We only need to predict for the current date using ALL history before it.
    
    history_df = pd.DataFrame()
    
    # We can't loop day by day if dataset is huge (8000 rows is fine though).
    
    for d in dates:
        # Fights on this date
        current_fights = df[df['event_date'] == d].copy()
        
        if len(history_df) > 100: # Need some history
            # Calculate priors from history
            # 1. Referee Priors
            ref_priors = fit_rest_priors(history_df, group_cols=['referee'], strat_cols=['weight_class'])
            if not ref_priors.empty:
                # Merge to current fights
                # We need to match on referee and weight_class
                # Rename col to avoid collision
                ref_priors = ref_priors.rename(columns={'REST_finish_mult': 'REST_ref_mult'})
                # Drop extra cols
                ref_priors = ref_priors[['referee', 'weight_class', 'REST_ref_mult']]
                
                current_fights = current_fights.merge(ref_priors, on=['referee', 'weight_class'], how='left')
            else:
                current_fights['REST_ref_mult'] = 1.0
                
            # 2. Location Priors (State)
            state_priors = fit_rest_priors(history_df, group_cols=['event_state'], strat_cols=['weight_class'])
            if not state_priors.empty:
                state_priors = state_priors.rename(columns={'REST_finish_mult': 'REST_state_mult'})
                state_priors = state_priors[['event_state', 'weight_class', 'REST_state_mult']]
                current_fights = current_fights.merge(state_priors, on=['event_state', 'weight_class'], how='left')
            else:
                current_fights['REST_state_mult'] = 1.0
                
            # 3. Country Priors
            country_priors = fit_rest_priors(history_df, group_cols=['event_country'], strat_cols=['weight_class'])
            if not country_priors.empty:
                country_priors = country_priors.rename(columns={'REST_finish_mult': 'REST_country_mult'})
                country_priors = country_priors[['event_country', 'weight_class', 'REST_country_mult']]
                current_fights = current_fights.merge(country_priors, on=['event_country', 'weight_class'], how='left')
            else:
                current_fights['REST_country_mult'] = 1.0
                
        else:
            current_fights['REST_ref_mult'] = 1.0
            current_fights['REST_state_mult'] = 1.0
            current_fights['REST_country_mult'] = 1.0
            
        # Store results
        # We only need fight_id and the new features
        feats = current_fights[['fight_id', 'REST_ref_mult', 'REST_state_mult', 'REST_country_mult']].copy()
        rest_features.append(feats)
        
        # Update history
        history_df = pd.concat([history_df, current_fights])
        
    return pd.concat(rest_features)

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Add fight_id if missing
    if 'fight_id' not in df.columns:
        df['fight_id'] = df.index
        
    # Load Golden for extra columns
    print("Loading golden data for REST columns...")
    try:
        golden = pd.read_csv('UFC_full_data_golden.csv')
        golden['event_date'] = pd.to_datetime(golden['event_date'])
        
        # Columns we need
        needed = ['event_date', 'f_1_name', 'f_2_name', 'event_state', 'event_country', 'result', 'outcome_method']
        # Check what exists
        needed = [c for c in needed if c in golden.columns]
        print(f"Columns to merge from golden: {needed}")
        
        # Merge
        # We match on date and names
        print(f"Pre-merge columns: {df.columns.tolist()[:10]}...")
        df = df.merge(golden[needed], on=['event_date', 'f_1_name', 'f_2_name'], how='left')
        print("Merged golden columns.")
        print(f"Post-merge columns: {df.columns.tolist()[:10]}...")
        
        # Handle potential suffixes if columns already existed (unlikely but possible)
        if 'result_y' in df.columns:
            df['result'] = df['result_y'].fillna(df.get('result_x'))
        
    except Exception as e:
        print(f"Failed to load golden: {e}")
        import traceback
        traceback.print_exc()
        return

    # Calculate REST
    rest_stats = calculate_rest_expanding(df)
    
    # Check if rest_stats is valid
    if rest_stats.empty or 'fight_id' not in rest_stats.columns:
        print("WARNING: REST calculation returned empty or invalid DataFrame.")
        # Create dummy features to allow script to continue (or fail gracefully)
        # But we want to see why it failed.
        # If it failed, we can't train.
        return
    print(f"Calculated REST stats for {len(rest_stats)} fights.")
    
    # Merge back
    # rest_stats has fight_id and features
    # Ensure no duplicates
    rest_stats = rest_stats.drop_duplicates(subset=['fight_id'])
    
    df = df.merge(rest_stats, on='fight_id', how='left')
    
    new_feats = ['REST_ref_mult', 'REST_state_mult', 'REST_country_mult']
    
    # Fill NA with 1.0 (neutral)
    for f in new_feats:
        df[f] = df[f].fillna(1.0)
        
    # Train & Evaluate
    print("Training model with REST features...")
    
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    features.extend(new_feats)
    features = [f for f in features if 'duration' not in f]
    
    # Encode categoricals
    from sklearn.preprocessing import LabelEncoder
    cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        
    # Target
    f1_wins = df['winner'] == df['f_1_name']
    f2_wins = df['winner'] == df['f_2_name']
    df = df[f1_wins | f2_wins].copy()
    df['target'] = (df['winner'] == df['f_1_name']).astype(int)
    
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df['target']
    X_test = test_df[features]
    y_test = test_df['target']
    
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        colsample_bytree=0.6,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )
    
    from sklearn.metrics import accuracy_score, log_loss
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    print(f"\n--- REST Model Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    
    # Save results
    with open('experimental/rest_results.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Log Loss: {ll:.4f}\n")
        
        imp = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        f.write("\nNew Feature Importance:\n")
        f.write(imp[imp['feature'].isin(new_feats)].to_string())

if __name__ == "__main__":
    run_test()
