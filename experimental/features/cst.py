import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.linear_model import LogisticRegression

def _standardize(X: np.ndarray):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    return (X-mu)/sd, mu, sd

def logistic_density_ratio(source_X: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    n = source_X.shape[0]
    X = np.vstack([source_X, np.repeat(target_x[None,:], n, axis=0)])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    ps = clf.predict_proba(source_X)[:,1]
    eps = 1e-6
    w = ps / np.clip(1-ps, eps, None)
    return w / (w.mean()+1e-12)

def transported_expectations(fighter_past: pd.DataFrame,
                             perf_cols: List[str],
                             opp_style_cols: List[str]) -> Dict[str,float]:
    # assert 'TARGET' in fighter_past.columns, "Provide TARGET col with one True row as upcoming opponent style."
    if 'TARGET' not in fighter_past.columns:
         return {}
         
    target_row = fighter_past.loc[fighter_past['TARGET']]
    if len(target_row) == 0:
        return {}
    target_row = target_row.iloc[0]
    
    target_x = target_row[opp_style_cols].to_numpy(float)
    src = fighter_past.loc[~fighter_past['TARGET']].copy()
    
    if len(src) < 5: # Need enough history
        return {}
        
    X = src[opp_style_cols].to_numpy(float)
    X_std, mu, sd = _standardize(X)
    tx_std = (target_x-mu)/sd
    w = logistic_density_ratio(X_std, tx_std)
    out = {}
    for c in perf_cols:
        vals = src[c].to_numpy(float)
        mean = np.sum(w*vals)/(w.sum()+1e-12)
        out[f"CST_{c}"] = float(mean)
        out[f"CST_{c}_var"] = float(np.average((vals-mean)**2, weights=w))
    out["CST_weight_n_eff"] = float((w.sum()**2)/(np.sum(w**2)+1e-12))
    return out
