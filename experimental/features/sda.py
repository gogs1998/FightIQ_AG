import numpy as np
import pandas as pd
from typing import List

def _prob(v, eps=1e-12):
    v = np.asarray(v, float)
    v = np.clip(v, eps, None)
    return v / v.sum()

def entropy(p):
    p = _prob(p)
    return float(-(p*np.log(p)).sum())

def js_div(p, q):
    p, q = _prob(p), _prob(q)
    m = 0.5*(p+q)
    return float(0.5*((p*np.log(p/m)).sum() + (q*np.log(q/m)).sum()))

def compute_sda(df: pd.DataFrame, mix_cols: List[str], fighter_col='fighter_id', time_col='date') -> pd.DataFrame:
    df = df.copy().sort_values([fighter_col, time_col])
    rows = []
    for fid, g in df.groupby(fighter_col):
        if len(g) < 2: continue
        
        mixes = g[mix_cols].to_numpy(float)
        ent = [entropy(m) for m in mixes]
        
        # JS div between consecutive fights
        js = [js_div(mixes[i-1], mixes[i]) for i in range(1,len(mixes))] if len(mixes)>1 else [np.nan]
        
        # We want the stats *entering* the next fight?
        # The function returns one row per fighter summarizing history?
        # Or should it return a time-series?
        # The original code returns one row per fighter:
        # 'SDA_entropy_mean', 'SDA_js_median'
        # This implies it's a static feature for the fighter's *whole career*?
        # That would be leakage if we include the current fight.
        
        # Let's modify it to be expanding window like PEAR if possible, 
        # OR we just use it as "past history" summary.
        
        # For this implementation, let's return the summary stats based on the provided dataframe.
        # The caller must ensure the dataframe only contains PAST fights.
        
        rows.append({'fighter_id': fid,
                     'SDA_entropy_mean': float(np.nanmean(ent)),
                     'SDA_js_median': float(np.nanmedian(js)),
                     'n_fights': int(len(g))})
    return pd.DataFrame(rows)
