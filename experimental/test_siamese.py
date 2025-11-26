import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Add current dir to path
sys.path.append(os.path.join(os.getcwd(), 'experimental'))
from models.siamese import SiameseMatchupNet, symmetric_loss

def prepare_siamese_data(df, features):
    """
    Split features into fighter A and fighter B sets.
    """
    # Load safe features
    with open('features_elo.json', 'r') as f:
        safe_features = set(json.load(f))
        
    # Load XGBoost model to get feature importance
    import joblib
    try:
        xgb_model = joblib.load('ufc_model_elo.pkl')
        # Get importance
        # We need the feature names list from json to map index to name
        with open('features_elo.json', 'r') as f:
            xgb_feat_names = json.load(f)
            
        importances = xgb_model.feature_importances_
        feat_imp = dict(zip(xgb_feat_names, importances))
        
        # Select Top N features
        TOP_N = 50
        sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        top_features = set([x[0] for x in sorted_feats[:TOP_N]])
        print(f"Selected Top {TOP_N} features based on XGBoost importance.")
        
    except Exception as e:
        print(f"Could not load XGBoost model for importance: {e}")
        top_features = safe_features # Fallback to all
    
    # Identify pairs from safe features
    # We want strict (f1_col, f2_col) tuples.
    
    pairs = set() # Store (f1, f2) tuples
    all_cols = set(df.columns)
    
    for feat in safe_features:
        # FILTER: Only consider this feature if it (or its diff/base) is in top_features
        # But wait, 'diff_elo' might be important, so we want 'f_1_elo' and 'f_2_elo'.
        # So if 'diff_X' is important, we want X pairs.
        # If 'f_1_X' is important, we want X pairs.
        
        is_important = False
        if feat in top_features:
            is_important = True
        
        # We proceed to find the pair, but we only add it if it's related to an important feature.
        
        base = None
        f1_col = None
        f2_col = None
        
        if feat.startswith('diff_'):
            base = feat[5:] # remove diff_
            # Try prefix
            if f"f_1_{base}" in all_cols and f"f_2_{base}" in all_cols:
                f1_col = f"f_1_{base}"
                f2_col = f"f_2_{base}"
            # Try suffix
            elif f"{base}_f_1" in all_cols and f"{base}_f_2" in all_cols:
                f1_col = f"{base}_f_1"
                f2_col = f"{base}_f_2"
                
        elif '_f_1' in feat or 'f_1_' in feat:
            if feat.startswith('f_1_'):
                f1_col = feat
                f2_col = feat.replace('f_1_', 'f_2_')
            else:
                f1_col = feat
                f2_col = feat.replace('_f_1', '_f_2')
            if f2_col not in all_cols: f1_col = None
            
        elif '_f_2' in feat or 'f_2_' in feat:
            if feat.startswith('f_2_'):
                f2_col = feat
                f1_col = feat.replace('f_2_', 'f_1_')
            else:
                f2_col = feat
                f1_col = feat.replace('_f_2', '_f_1')
            if f1_col not in all_cols: f1_col = None
            
        if f1_col and f2_col and is_important:
            if (f1_col, f2_col) not in pairs:
                pairs.add((f1_col, f2_col))

    pairs = list(pairs)
    print(f"Found {len(pairs)} feature pairs related to Top {TOP_N} XGBoost features.")
    
    # Filter numeric only
    numeric_pairs = []
    for c1, c2 in pairs:
        if pd.api.types.is_numeric_dtype(df[c1]) and pd.api.types.is_numeric_dtype(df[c2]):
            numeric_pairs.append((c1, c2))
            
    print(f"Filtered to {len(numeric_pairs)} available numeric pairs.")
    # print(f"Pairs used: {numeric_pairs}")
    
    f1_feats = [p[0] for p in numeric_pairs]
    f2_feats = [p[1] for p in numeric_pairs]
    
    # Extract data
    # We need to handle NaNs.
    X1 = df[f1_feats].copy()
    X2 = df[f2_feats].copy()
    
    # Rename columns to generic names for shared scaling
    # We strip 'f_1_' and 'f_2_' prefixes/suffixes to get a common name
    # This is important for the scaler to treat them as the same "feature"
    
    generic_names = []
    for c in f1_feats:
        if c.startswith('f_1_'):
            generic_names.append(c[4:])
        elif c.endswith('_f_1'):
            generic_names.append(c[:-4])
        else:
            generic_names.append(c) # Should not happen given logic above
            
    X1.columns = generic_names
    X2.columns = generic_names
    
    # Fill NA
    X1 = X1.fillna(0) 
    X2 = X2.fillna(0)
    
    # Scale
    scaler = StandardScaler()
    
    # Stack to fit scaler
    combined = pd.concat([X1, X2], axis=0)
    scaler.fit(combined)
    
    X1_scaled = scaler.transform(X1)
    X2_scaled = scaler.transform(X2)
    
    y = (df['winner'] == df['f_1_name']).astype(float).to_numpy()
    
    return X1_scaled, X2_scaled, y, f1_feats, f2_feats

def run_test():
    print("Loading data...")
    df = pd.read_csv('UFC_data_with_elo.csv')
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values('event_date')
    
    # Filter finished fights
    df = df[df['winner'].notna()]
    
    # Prepare data
    print("Preparing data for Siamese Net...")
    X1, X2, y, f1_names, f2_names = prepare_siamese_data(df, None)
    
    # Split time-based
    split_idx = int(len(df) * 0.85)
    
    X1_train, X1_test = X1[:split_idx], X1[split_idx:]
    X2_train, X2_test = X2[:split_idx], X2[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to tensors
    t_X1_train = torch.tensor(X1_train, dtype=torch.float32)
    t_X2_train = torch.tensor(X2_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32)
    
    t_X1_test = torch.tensor(X1_test, dtype=torch.float32)
    t_X2_test = torch.tensor(X2_test, dtype=torch.float32)
    t_y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Dataset
    train_ds = TensorDataset(t_X1_train, t_X2_train, t_y_train)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    # Model
    in_dim = X1.shape[1]
    print(f"Input dimension: {in_dim}")
    
    model = SiameseMatchupNet(in_dim=in_dim, hidden=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("Training Siamese Net...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for b_x1, b_x2, b_y in train_loader:
            optimizer.zero_grad()
            
            # Loss
            loss = symmetric_loss(model, b_x1, b_x2, b_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
    # Evaluate Siamese
    model.eval()
    with torch.no_grad():
        siamese_probs = model(t_X1_test, t_X2_test).numpy()
        siamese_preds = (siamese_probs > 0.5).astype(int)
        
    acc = accuracy_score(y_test, siamese_preds)
    ll = log_loss(y_test, siamese_probs)
    
    print(f"\n--- Siamese Net Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")
    
    # --- ENSEMBLE ---
    print("\n--- Ensemble Evaluation ---")
    try:
        import joblib
        import xgboost as xgb
        
        # Load XGBoost model (pickled)
        xgb_model = joblib.load('ufc_model_elo.pkl')
        
        # Load features for XGBoost
        with open('features_elo.json', 'r') as f:
            xgb_features = json.load(f)
            
        # Prepare Test Data for XGBoost
        # We need to ensure encoding matches training.
        # In train_elo.py, encoding was done on the full DF before split.
        # We should have done that at the start.
        # But since we didn't, let's do it now on a copy of the full df, then split.
        
        df_xgb = df.copy()
        
        # Handle Categoricals (Naive Label Encoding as in train_elo.py)
        cat_cols = df_xgb[xgb_features].select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            df_xgb[col] = df_xgb[col].astype(str)
            df_xgb[col] = le.fit_transform(df_xgb[col])
            
        # Split
        test_df_xgb = df_xgb.iloc[split_idx:]
        X_test_xgb = test_df_xgb[xgb_features]
        
        # Predict
        xgb_probs = xgb_model.predict_proba(X_test_xgb)[:, 1]
        xgb_acc = accuracy_score(y_test, (xgb_probs > 0.5).astype(int))
        xgb_ll = log_loss(y_test, xgb_probs)
        
        print(f"XGBoost Baseline -> Accuracy: {xgb_acc:.4f}, Log Loss: {xgb_ll:.4f}")
        
        # Ensemble (Average)
        ens_probs = 0.5 * siamese_probs + 0.5 * xgb_probs
        ens_preds = (ens_probs > 0.5).astype(int)
        
        ens_acc = accuracy_score(y_test, ens_preds)
        ens_ll = log_loss(y_test, ens_probs)
        
        print(f"Ensemble (50/50) -> Accuracy: {ens_acc:.4f}, Log Loss: {ens_ll:.4f}")
        
        # Save results
        with open('experimental/siamese_results.txt', 'w') as f:
            f.write(f"Siamese Accuracy: {acc:.4f}\n")
            f.write(f"Siamese Log Loss: {ll:.4f}\n")
            f.write(f"XGBoost Accuracy: {xgb_acc:.4f}\n")
            f.write(f"XGBoost Log Loss: {xgb_ll:.4f}\n")
            f.write(f"Ensemble Accuracy: {ens_acc:.4f}\n")
            f.write(f"Ensemble Log Loss: {ens_ll:.4f}\n")
            
    except Exception as e:
        print(f"Ensemble failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
