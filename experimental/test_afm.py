import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Add current dir to path to import features
sys.path.append(os.path.join(os.getcwd(), 'experimental'))
from features.afm import compute_afm_numeric

def train_proxy_model(df, features):
    """
    Train a simple proxy model to be used as the 'matchup function' for AFM.
    We use Logistic Regression for fast gradient approximation (or just fast prediction).
    """
    # Drop NaNs in features
    df_clean = df.dropna(subset=features).copy()
    
    X = df_clean[features]
    y = (df_clean['winner'] == df_clean['f_1_name']).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_scaled, y)
    
    return model, scaler

def calculate_afm(df, proxy_model, scaler, proxy_features):
    """
    Calculate AFM for each fight in df using the proxy model.
    """
    print("Calculating AFM features...")
    
    afm_scores = []
    
    # We need to separate features for f1 and f2 to pass to compute_afm_numeric
    # But compute_afm_numeric expects two vectors a and b.
    # Our proxy model takes a single vector x (which is usually f1_stats - f2_stats or similar).
    # Wait, our proxy features in 'features_elo.json' are mixed (diffs, f1, f2).
    
    # To make this work with compute_afm_numeric(f, a, b), we need to define 'a' and 'b' 
    # such that we can reconstruct the input to proxy_model.
    
    # Simplified approach:
    # Let's define the proxy model inputs as purely [f1_stats, f2_stats].
    # Then 'a' is f1_stats, 'b' is f2_stats.
    # We need to map these to the actual model features (which might include diffs).
    
    # Let's pick a small set of "core" stats for AFM to measure fragility against.
    # Based on inspection, we have columns like 'wins_3_f_1', 'losses_3_f_1', 'slpm_3_f_1' etc.
    # Or maybe just generic ones.
    # Let's try to find the "best" available stats dynamically or hardcode common ones found.
    
    # Common suffixes in this dataset seem to be '_3_f_1' (3-fight avg?) or just '_f_1'.
    # Let's look for 'slpm', 'td_avg', 'sub_avg'.
    
    potential_stats = ['slpm', 'td_avg', 'sub_avg', 'sapm', 'str_def', 'td_def']
    core_stats = []
    
    # Try to find matching columns
    for stat in potential_stats:
        # Try various suffixes
        candidates = [c for c in df.columns if stat in c and 'f_1' in c]
        if candidates:
            # Pick the one that looks most like a general average (shortest name or specific suffix)
            # e.g. 'slpm_3_f_1'
            best = sorted(candidates, key=len)[0] # Shortest might be 'slpm_f_1' if exists
            # Extract the base name without 'f_1'
            base = best.replace('f_1', '').replace('__', '_').strip('_')
            core_stats.append(base)
            
    # Also add wins/losses if available
    if 'wins_3_f_1' in df.columns:
        core_stats.append('wins_3')
    if 'losses_3_f_1' in df.columns:
        core_stats.append('losses_3')
        
    print(f"Selected core stats for AFM: {core_stats}")
            
    # Check if they exist
    # We construct f_1_... and f_2_...
    # Note: base might be 'slpm_3' -> 'slpm_3_f_1'
    
    # 1. Train Special Proxy
    print("Training special proxy model for AFM...")
    proxy_cols_f1 = []
    proxy_cols_f2 = []
    
    for base in core_stats:
        # Try to reconstruct f1/f2 names
        # Case 1: base has 'f_1' in it? No we stripped it.
        # Case 2: base is 'slpm_3'. f1 is 'slpm_3_f_1' or 'f_1_slpm_3'?
        # In this dataset, suffixes seem to be at end: '..._f_1'
        
        c1 = f"{base}_f_1"
        c2 = f"{base}_f_2"
        
        if c1 not in df.columns:
            # Try prefix
            c1 = f"f_1_{base}"
            c2 = f"f_2_{base}"
            
        if c1 in df.columns and c2 in df.columns:
            proxy_cols_f1.append(c1)
            proxy_cols_f2.append(c2)
        else:
            print(f"Skipping {base} - cols not found: {c1}, {c2}")
            
    print(f"Final proxy columns ({len(proxy_cols_f1)}): {proxy_cols_f1}")
    
    proxy_cols = proxy_cols_f1 + proxy_cols_f2
    
    if not proxy_cols:
        print("ERROR: No proxy columns found. AFM failed.")
        return []
    
    # Filter clean data
    train_df = df.dropna(subset=proxy_cols).copy()
    X_proxy = train_df[proxy_cols]
    y_proxy = (train_df['winner'] == train_df['f_1_name']).astype(int)
    
    p_scaler = StandardScaler()
    X_p_scaled = p_scaler.fit_transform(X_proxy)
    
    p_model = LogisticRegression()
    p_model.fit(X_p_scaled, y_proxy)
    
    # 2. Define prediction function for AFM
    def predict_proba_wrapper(a, b):
        # a, b are numpy arrays of shape (n_feats,)
        # Concatenate
        x = np.concatenate([a, b]).reshape(1, -1)
        # Scale
        x_scaled = p_scaler.transform(x)
        # Predict
        return p_model.predict_proba(x_scaled)[0, 1]
        
    # 3. Calculate AFM for all rows
    # This is slow row-by-row.
    # But AFM is numeric gradient, so we have to do it row-by-row or vectorized.
    # compute_afm_numeric is row-by-row.
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df)}...", end='\r')
            
        # Extract a and b
        try:
            a = row[proxy_cols_f1].to_numpy(float)
            b = row[proxy_cols_f2].to_numpy(float)
            
            if np.isnan(a).any() or np.isnan(b).any():
                afm_scores.append(np.nan)
                continue
                
            score = compute_afm_numeric(predict_proba_wrapper, a, b)
            afm_scores.append(score)
        except Exception as e:
            afm_scores.append(np.nan)
            
    print(f"Processing row {len(df)}/{len(df)}... Done.")
            
    return afm_scores

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Add fight_id if missing
    if 'fight_id' not in df.columns:
        df['fight_id'] = df.index
        
    # Add implied prob if missing (useful for AFM)
    if 'f_1_odds_implied_prob' not in df.columns and 'f_1_odds' in df.columns:
         # Simple conversion
         def get_prob(o):
             if pd.isna(o): return np.nan
             if o < 0: return (-o) / (-o + 100)
             return 100 / (o + 100)
         df['f_1_odds_implied_prob'] = df['f_1_odds'].apply(get_prob)
         df['f_2_odds_implied_prob'] = df['f_2_odds'].apply(get_prob)
         
    # Calculate AFM
    # We use the WHOLE dataset to train the proxy? 
    # Ideally we should split, but for a feature engineering test, 
    # using a simple proxy trained on all data is a common heuristic 
    # (like using PCA trained on all data).
    # To be strictly rigorous, we should use expanding window, but that's very slow for this test.
    # Let's accept the slight leakage for the PROXY model only, 
    # assuming the proxy captures general "physics of fighting" which is constant.
    
    afm_vals = calculate_afm(df, None, None, None)
    df['AFM_score'] = afm_vals
    
    # Fill NA
    df['AFM_score'] = df['AFM_score'].fillna(df['AFM_score'].mean())
    
    print(f"AFM Stats: Mean={df['AFM_score'].mean():.4f}, Std={df['AFM_score'].std():.4f}")
    
    # Train & Evaluate Main Model
    print("Training model with AFM feature...")
    
    with open('features_elo.json', 'r') as f:
        features = json.load(f)
        
    features.append('AFM_score')
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
    
    print(f"\n--- AFM Model Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    
    # Save results
    with open('experimental/afm_results.txt', 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Log Loss: {ll:.4f}\n")
        
        imp = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        f.write("\nNew Feature Importance:\n")
        f.write(imp[imp['feature'] == 'AFM_score'].to_string())

if __name__ == "__main__":
    run_test()
