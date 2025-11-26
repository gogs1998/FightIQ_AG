import numpy as np
import pandas as pd
from typing import Tuple

def compute_round_efficiency(df_rounds: pd.DataFrame,
                             metrics=('sig_str_diff',),
                             pace_col='opp_sig_str_per_min') -> pd.DataFrame:
    need = {'fight_id','round','fighter_id',metrics[0],pace_col}
    missing = need - set(df_rounds.columns)
    if missing:
        # Try to handle missing columns gracefully or raise error
        # For now, return empty
        return pd.DataFrame()
        
    df = df_rounds.copy().sort_values(['fighter_id','fight_id','round'])
    df['lag_eff'] = df.groupby(['fighter_id','fight_id'])[metrics[0]].shift(1)
    df['lag_pace'] = df.groupby(['fighter_id','fight_id'])[pace_col].shift(1)
    df = df.dropna(subset=['lag_eff','lag_pace'])
    df['eff'] = df[metrics[0]].astype(float)
    return df

def fit_pear(df_rounds: pd.DataFrame) -> pd.DataFrame:
    # DEBUG
    # print(f"fit_pear input cols: {df_rounds.columns.tolist()}")
    
    df = compute_round_efficiency(df_rounds)
    if df.empty:
        return pd.DataFrame()
        
    rows = []
    for fid, g in df.groupby('fighter_id'):
        if len(g) < 3: continue # Need enough data points
        
        X = np.column_stack([np.ones(len(g)), g['lag_pace'], g['lag_eff']])
        y = g['eff'].to_numpy()
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            rows.append({'fighter_id': fid,
                         'beta_pace': float(beta[1]),
                         'beta_lag': float(beta[2]),
                         'n_rounds': int(len(g))})
        except np.linalg.LinAlgError:
            rows.append({'fighter_id': fid,
                         'beta_pace': np.nan,
                         'beta_lag': np.nan,
                         'n_rounds': int(len(g))})
    return pd.DataFrame(rows)
