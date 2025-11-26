import pandas as pd
import numpy as np
import joblib
import json
import sys
import os

# Add current dir to path to import features
sys.path.append(os.path.join(os.getcwd(), 'experimental'))
from features.cst import transported_expectations

def calculate_cst_expanding(df):
    """
    Calculate CST stats for each fight using only PAST history.
    """
    df = df.sort_values('event_date')
    
    # Add fight_id if missing
    if 'fight_id' not in df.columns:
        df['fight_id'] = df.index
        
    # Define Style Columns (Opponent Attributes)
    # We need columns that describe the opponent's style.
    # These should be available for the opponent *before* the fight.
    # In our dataset, 'f_2_avg_sig_str_landed', 'f_2_height_cms', etc. are good candidates.
    # But we need to be careful about names.
    # For a past fight (A vs C), we want C's stats.
    
    style_cols_template = [
        'avg_sig_str_landed',
        'avg_td_landed',
        'avg_sub_att',
        'height_cms',
        'reach_cms'
    ]
    
    # Performance Columns (What we want to predict for A)
    # These are A's stats in the fight.
    # In the dataset, for A vs C, if A is f_1, these are 'f_1_sig_strikes_landed' etc.
    # BUT we need the raw fight stats, not averages.
    # We'll use the same proxy as PEAR: 'f_1_sig_strikes_landed' (if available) or similar.
    # Wait, 'f_1_sig_strikes_landed' IS available in the raw data (UFC_full_data_golden), 
    # but maybe not in 'UFC_data_with_elo.csv' if we dropped them?
    # We found 'f_1_r1_sig_strikes' earlier.
    # Let's assume we have access to 'f_1_sig_strikes_landed' (total) or we sum rounds.
    
    # Actually, let's check columns first.
    # If not available, we can't do CST on performance metrics.
    # We could try to transport the 'winner' (0/1) but that's classification.
    # CST is usually for regression of performance metrics.
    
    # Let's assume we can find: 'f_1_sig_strikes_landed', 'f_1_takedowns_landed'
    perf_cols_template = ['sig_strikes_landed', 'takedowns_landed']
    
    cst_features = []
    history = {} # fighter_id -> list of dicts (past fights)
    
    print("Calculating CST features (this is slow)...")
    
    # We need to process every fight.
    # For A vs B:
    # 1. Get A's history (A vs C, A vs D...)
    # 2. Create a DF where each row is a past opponent (C, D...)
    #    - Columns: Style of C, Style of D...
    #    - Target Col: Performance of A vs C, A vs D...
    # 3. Create a TARGET row for B
    #    - Columns: Style of B
    # 4. Run CST
    
    # To make this efficient, we need to store the style/perf data in history.
    
    for idx, row in df.iterrows():
        fid = row['fight_id']
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        
        # Check if we have necessary columns for this row (to build history)
        # We need f_1_style (for f2's history), f_2_style (for f1's history)
        # And f_1_perf, f_2_perf
        
        # Map template to actual columns
        # Style of F1 (to be stored in F2's history as opponent)
        f1_style = {}
        for c in style_cols_template:
            col_name = f'f_1_{c}'
            # If not found, try without 'avg_' or other variations?
            # For now assume exact match or handle missing
            f1_style[c] = row.get(col_name, 0)
            
        f2_style = {}
        for c in style_cols_template:
            col_name = f'f_2_{c}'
            f2_style[c] = row.get(col_name, 0)
            
        # Performance of F1
        f1_perf = {}
        for c in perf_cols_template:
            col_name = f'f_1_{c}' # e.g. f_1_sig_strikes_landed
            # If not in df, maybe we can sum rounds?
            # Or maybe it IS in df (we should check).
            # For now, use get with 0 default, but this is risky if it's always 0.
            f1_perf[c] = row.get(col_name, 0)
            
        f2_perf = {}
        for c in perf_cols_template:
            col_name = f'f_2_{c}'
            f2_perf[c] = row.get(col_name, 0)
            
        # --- CST Calculation ---
        
        # F1 CST (Transport A's history to B)
        # A's history contains opponents (C, D...) with their styles and A's perf against them.
        # Target is B's style.
        
        if f1 in history and len(history[f1]) >= 5:
            # Construct DF
            # History items: {'opp_style': {...}, 'my_perf': {...}}
            # Flatten for DataFrame
            data = []
            for h in history[f1]:
                item = {}
                for k, v in h['opp_style'].items(): item[k] = v
                for k, v in h['my_perf'].items(): item[k] = v
                item['TARGET'] = False
                data.append(item)
            
            # Add Target (B)
            target_item = {}
            for k, v in f2_style.items(): target_item[k] = v
            # Perf cols for target are unknown (we want to predict them), fill with NaN or 0
            for k in perf_cols_template: target_item[k] = 0 
            target_item['TARGET'] = True
            data.append(target_item)
            
            hist_df = pd.DataFrame(data)
            
            # Run CST
            try:
                res = transported_expectations(hist_df, perf_cols_template, style_cols_template)
                f1_cst = res
            except Exception as e:
                # print(f"CST Error: {e}")
                f1_cst = {}
        else:
            f1_cst = {}
            
        # F2 CST (Transport B's history to A)
        if f2 in history and len(history[f2]) >= 5:
            data = []
            for h in history[f2]:
                item = {}
                for k, v in h['opp_style'].items(): item[k] = v
                for k, v in h['my_perf'].items(): item[k] = v
                item['TARGET'] = False
                data.append(item)
            
            target_item = {}
            for k, v in f1_style.items(): target_item[k] = v
            for k in perf_cols_template: target_item[k] = 0
            target_item['TARGET'] = True
            data.append(target_item)
            
            hist_df = pd.DataFrame(data)
            
            try:
                res = transported_expectations(hist_df, perf_cols_template, style_cols_template)
                f2_cst = res
            except:
                f2_cst = {}
        else:
            f2_cst = {}
            
        # Store Features
        feat = {'fight_id': fid}
        for k, v in f1_cst.items(): feat[f'f_1_{k}'] = v
        for k, v in f2_cst.items(): feat[f'f_2_{k}'] = v
        cst_features.append(feat)
        
        # --- Update History ---
        # Add this fight to history
        if f1 not in history: history[f1] = []
        history[f1].append({'opp_style': f2_style, 'my_perf': f1_perf})
        
        if f2 not in history: history[f2] = []
        history[f2].append({'opp_style': f1_style, 'my_perf': f2_perf})
        
    return pd.DataFrame(cst_features)

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Add fight_id if missing
    if 'fight_id' not in df.columns:
        df['fight_id'] = df.index
        
    # Check if we have the columns we need
    # We need 'f_1_avg_sig_str_landed' etc.
    # And 'f_1_sig_strikes_landed' (raw)
    
    print("Checking columns...")
    cols = df.columns.tolist()
    req_style = ['avg_sig_str_landed', 'avg_td_landed', 'height_cms'] # minimal set
    req_perf = ['sig_strikes_landed'] # minimal set
    
    missing = []
    for c in req_style:
        if f'f_1_{c}' not in cols: missing.append(f'f_1_{c}')
        
    # If raw perf cols are missing, we might need to load from golden or skip
    # Let's check specifically for 'f_1_sig_strikes_landed'
    if 'f_1_sig_strikes_landed' not in cols:
        print("WARNING: Raw performance columns missing. CST requires raw fight stats to build history.")
        print("Attempting to load from golden...")
        try:
            golden = pd.read_csv('UFC_full_data_golden.csv')
            # Merge raw stats
            # We need to match on something unique. Date + Names?
            # Or just assume same order if we sort? Risky.
            # Let's try to merge on event_date, f_1_name, f_2_name
            golden['event_date'] = pd.to_datetime(golden['event_date'])
            
            # Select only needed cols
            needed = ['event_date', 'f_1_name', 'f_2_name', 'f_1_sig_strikes_landed', 'f_2_sig_strikes_landed', 'f_1_takedowns_landed', 'f_2_takedowns_landed']
            # Check if they exist in golden
            needed = [c for c in needed if c in golden.columns]
            
            df = df.merge(golden[needed], on=['event_date', 'f_1_name', 'f_2_name'], how='left')
            print("Merged raw stats from golden.")
        except Exception as e:
            print(f"Failed to load golden: {e}")
            return

    # Calculate CST
    cst_stats = calculate_cst_expanding(df)
    print(f"Calculated CST stats for {len(cst_stats)} fights.")
    
    # Merge
    df = df.merge(cst_stats, on='fight_id', how='left')
    
    # Identify new features
    new_feats = [c for c in cst_stats.columns if c != 'fight_id']
    print(f"New Features: {new_feats}")
    
    # Fill NA
    for f in new_feats:
        df[f] = df[f].fillna(0) # 0 means "average" in standardized space? No, 0 means "no prediction".
        
    # Train & Evaluate
    print("Training model with CST features...")
    
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
    
    print(f"\n--- CST Model Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    
    # Save results
    with open('experimental/cst_results.txt', 'w') as f:
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
