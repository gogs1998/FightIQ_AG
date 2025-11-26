import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

def expanding_window_split(dates, n_splits=5):
    """
    Yields (train_idx, test_idx) pairs for expanding window CV.
    Train is always strictly before Test.
    """
    dates = pd.to_datetime(dates)
    unique_dates = np.sort(dates.unique())
    
    # We need at least n_splits + 1 chunks to have a train set for the first fold
    if len(unique_dates) < n_splits + 1:
        # Fallback: just split by count if dates are too sparse
        n = len(dates)
        indices = np.arange(n)
        chunk_size = n // (n_splits + 1)
        for i in range(n_splits):
            train_end = (i + 1) * chunk_size
            test_end = (i + 2) * chunk_size if i < n_splits - 1 else n
            yield indices[:train_end], indices[train_end:test_end]
        return

    # Time-based split
    # Divide unique dates into n_splits + 1 chunks
    chunk_size = len(unique_dates) // (n_splits + 1)
    
    for i in range(n_splits):
        # Train on 0..i chunks
        # Test on i+1 chunk
        
        train_date_end_idx = (i + 1) * chunk_size
        test_date_end_idx = (i + 2) * chunk_size if i < n_splits - 1 else len(unique_dates)
        
        train_cutoff = unique_dates[train_date_end_idx]
        test_cutoff = unique_dates[test_date_end_idx-1] # inclusive of the chunk
        
        # Actually, simpler:
        # Split unique dates into N blocks.
        # Fold 1: Train [Block 0], Test [Block 1]
        # Fold 2: Train [Block 0-1], Test [Block 2]
        # ...
        
        split_points = np.array_split(unique_dates, n_splits + 1)
        
        train_dates = np.concatenate(split_points[:i+1])
        test_dates = split_points[i+1]
        
        # Find indices
        # This is slow if done naively. 
        # Assuming dates is aligned with X, y
        
        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)
        
        yield np.where(train_mask)[0], np.where(test_mask)[0]

def generate_oof_preds(models, X, y, dates, n_splits=5):
    """
    Generate Out-Of-Fold predictions for the training set using expanding window.
    models: dict of {name: model_instance}
    """
    # Initialize OOF arrays with NaN
    oof_preds = {name: np.full(len(y), np.nan) for name in models.keys()}
    
    print(f"Generating OOF predictions with {n_splits} splits...")
    
    for fold, (train_idx, test_idx) in enumerate(expanding_window_split(dates, n_splits)):
        print(f"  Fold {fold+1}/{n_splits}: Train size {len(train_idx)}, Test size {len(test_idx)}")
        
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te = X.iloc[test_idx]
        
        for name, model in models.items():
            # Clone to avoid mutating original
            clf = clone(model)
            
            # Handle sample weights if needed (not implemented here for simplicity)
            if len(np.unique(y_tr)) < 2:
                print(f"    Skipping {name} for fold {fold+1}: Only 1 class in train set.")
                continue
                
            clf.fit(X_tr, y_tr)
            
            # Predict
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(X_te)[:, 1]
            else:
                p = clf.predict(X_te)
                
            oof_preds[name][test_idx] = p
            
    return pd.DataFrame(oof_preds)

def fit_stack(X_meta, y):
    """
    Train meta-learner on OOF predictions.
    """
    # Drop rows with NaNs (early folds where we didn't have predictions)
    mask = ~X_meta.isna().any(axis=1)
    X_clean = X_meta[mask]
    y_clean = y.values[mask]
    
    print(f"Training Meta-Learner on {len(X_clean)} samples (dropped {len(y) - len(y_clean)} early samples).")
    print(f"y_clean value counts:\n{pd.Series(y_clean).value_counts()}")
    
    if len(X_clean) == 0:
        raise ValueError("No samples left for meta-learner after dropping NaNs (OOF generation failed?)")
        
    if len(np.unique(y_clean)) < 2:
        raise ValueError(f"y_clean has only 1 class: {np.unique(y_clean)}")
    
    meta_model = LogisticRegression()
    meta_model.fit(X_clean, y_clean)
    
    return meta_model
