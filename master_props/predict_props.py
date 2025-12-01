import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb

def predict_props():
    print("=== Prop Hunter: Generating Prop Bets ===")
    
    # 1. Load Data (2024-2025 Test Set)
    BASE_DIR = 'd:/AntiGravity/FightIQ/master'
    try:
        df = pd.read_csv(f'{BASE_DIR}/data/training_data.csv')
    except:
        df = pd.read_csv('d:/AntiGravity/FightIQ/training_data.csv')
        
    # Load Features
    with open(f'{BASE_DIR}/prop_hunter/features.json', 'r') as f:
        features = json.load(f)['confirmed']
        
    # Load Models
    print("Loading models...")
    # We need a Win Model. We'll train one on the fly or load one?
    # Ideally we'd load 'model_win.pkl' but we haven't saved the Optimized Boruta Win Model yet.
    # Let's quickly train the Win Model here (fast enough) or assume we saved it.
    # I'll train it quickly to be safe.
    
    # Filter valid odds
    has_odds = (df['f_1_odds'].notna()) & (df['f_1_odds'] > 1.0) & \
               (df['f_2_odds'].notna()) & (df['f_2_odds'] > 1.0)
    df = df[has_odds].copy()
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    X = df[[c for c in features if c in df.columns]].fillna(0)
    y = df['target'].values
    
    # Split
    split_date = '2024-01-01'
    mask_train = df['event_date'] < split_date
    mask_test = df['event_date'] >= split_date
    
    X_train = X[mask_train]
    X_test = X[mask_test]
    y_train = y[mask_train]
    
    # Train Win Model
    with open(f'{BASE_DIR}/experiment_2/boruta_params.json', 'r') as f:
        params = json.load(f)
    params['eval_metric'] = 'logloss'
    params['use_label_encoder'] = False
    
    model_win = xgb.XGBClassifier(**params)
    model_win.fit(X_train, y_train)
    
    # Load Prop Models
    model_finish = joblib.load(f'{BASE_DIR}/prop_hunter/model_finish.pkl')
    model_method = joblib.load(f'{BASE_DIR}/prop_hunter/model_method.pkl')
    model_round = joblib.load(f'{BASE_DIR}/prop_hunter/model_round.pkl')
    
    # 2. Predict
    print("Generating predictions...")
    p_win = model_win.predict_proba(X_test)[:, 1]
    p_finish = model_finish.predict_proba(X_test)[:, 1] # Prob of Finish
    p_ko = model_method.predict_proba(X_test)[:, 1] # Prob of KO (given finish)
    p_rounds = model_round.predict_proba(X_test) # [R1, R2, R3, R4, R5] (given finish)
    
    # 3. Combine & Display
    test_df = df[mask_test].copy()
    results = []
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # Base Probs
        pw = p_win[i]
        pf = p_finish[i]
        p_dec = 1 - pf
        pk = p_ko[i]
        ps = 1 - pk
        pr = p_rounds[i]
        
        # Derived Probs (Fighter 1)
        # P(F1 Win) = pw
        # P(F1 Finish) = pw * pf
        # P(F1 KO) = pw * pf * pk
        # P(F1 Sub) = pw * pf * ps
        # P(F1 Dec) = pw * p_dec
        # P(F1 R1) = pw * pf * pr[0]
        
        # We can do the same for Fighter 2 (assuming symmetry approx)
        # P(F2 Win) = 1 - pw
        
        f1 = row['f_1_name']
        f2 = row['f_2_name']
        
        # Let's just output F1 props for brevity
        res = {
            "Fighter": f1,
            "Opponent": f2,
            "P(Win)": pw,
            "P(Finish)": pf,
            "P(KO|Fin)": pk,
            "P(Sub|Fin)": ps,
            "P(F1 KO)": pw * pf * pk,
            "P(F1 Sub)": pw * pf * ps,
            "P(F1 Dec)": pw * p_dec,
            "P(F1 R1)": pw * pf * pr[0],
            "P(F1 R2)": pw * pf * pr[1],
            "P(F1 R3)": pw * pf * pr[2]
        }
        results.append(res)
        
    # Show Top 10 "Most Likely KOs"
    res_df = pd.DataFrame(results)
    top_kos = res_df.sort_values('P(F1 KO)', ascending=False).head(10)
    
    print("\n=== Top 10 Predicted KOs (2024-2025) ===")
    print(top_kos[['Fighter', 'Opponent', 'P(F1 KO)', 'P(Win)', 'P(Finish)', 'P(KO|Fin)']].to_string(index=False))
    
    # Show Top 10 "Most Likely Subs"
    top_subs = res_df.sort_values('P(F1 Sub)', ascending=False).head(10)
    print("\n=== Top 10 Predicted Subs (2024-2025) ===")
    print(top_subs[['Fighter', 'Opponent', 'P(F1 Sub)', 'P(Win)', 'P(Finish)', 'P(Sub|Fin)']].to_string(index=False))

if __name__ == "__main__":
    predict_props()
