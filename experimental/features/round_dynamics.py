"""Round dynamics feature engineering helpers.

All functions assume you pass **only historical fights up to a given cutoff**
for each fighter when building pre-fight features – that keeps everything leak-free.

Recommended minimal columns in round_df:
- fighter_id
- opponent_id
- fight_id
- fight_date (datetime64)
- round (int)
- pace (e.g. sig_strikes_per_min)
- accuracy (0-1)
- ctrl_share (0-1)
"""
import numpy as np
import pandas as pd
from typing import List, Dict

def _poly_fit_rounds(rounds: np.ndarray, values: np.ndarray, deg: int = 2):
    """Fit y ~ poly(round) of given degree. Returns coefficients (lowest degree first)."""
    if len(rounds) < deg+1:
        return np.array([np.nan]*(deg+1))
    coeffs = np.polyfit(rounds, values, deg=deg)
    # np.polyfit returns highest degree first; reverse for convenience
    return coeffs[::-1]

def build_behavioural_profiles(round_df: pd.DataFrame,
                               fighter_id_col: str = 'fighter_id',
                               fight_id_col: str = 'fight_id',
                               date_col: str = 'fight_date',
                               round_col: str = 'round',
                               pace_col: str = 'pace',
                               ctrl_col: str = 'ctrl_share') -> pd.DataFrame:
    """Compute fighter-level behaviour from historical rounds:
    - avg R1 pace, R3/R1 ratio
    - avg control slope across rounds
    - variability of pace & control.

    Returns one row per fighter summarising **all rows in round_df**.
    For pre-fight usage, call this on a table filtered to fights strictly before the prediction date.
    """
    df = round_df.copy()
    # Ensure columns exist
    for col in [fighter_id_col, fight_id_col, date_col, round_col, pace_col, ctrl_col]:
        if col not in df.columns:
            # If optional columns missing, fill with nan
            if col in [pace_col, ctrl_col]:
                df[col] = np.nan
            else:
                raise ValueError(f"Missing required column: {col}")

    rows = []
    for fid, g in df.groupby(fighter_id_col):
        # per-fight aggregates
        by_fight = []
        for _, gf in g.groupby(fight_id_col):
            gf = gf.sort_values(round_col)
            r = gf[round_col].to_numpy()
            pace = gf[pace_col].to_numpy(float)
            ctrl = gf[ctrl_col].to_numpy(float)
            
            # R1 Pace
            r1_pace = pace[r==1].mean() if (r==1).any() else np.nan
            
            # R3 Pace (or last round if < 3? No, specifically R3 for cardio check)
            r3_pace = pace[r==3].mean() if (r==3).any() else np.nan
            
            # Ratio
            ratio_r3_r1 = r3_pace / r1_pace if (r1_pace and not np.isnan(r1_pace) and r1_pace > 0) else np.nan
            
            # simple linear slope pace ~ round
            if len(r) >= 2:
                A = np.column_stack([np.ones(len(r)), r])
                # Check for NaNs
                mask = ~np.isnan(pace)
                if mask.sum() >= 2:
                    beta, *_ = np.linalg.lstsq(A[mask], pace[mask], rcond=None)
                    slope_pace = beta[1]
                else:
                    slope_pace = np.nan
                    
                mask_ctrl = ~np.isnan(ctrl)
                if mask_ctrl.sum() >= 2:
                    beta2, *_ = np.linalg.lstsq(A[mask_ctrl], ctrl[mask_ctrl], rcond=None)
                    slope_ctrl = beta2[1]
                else:
                    slope_ctrl = np.nan
            else:
                slope_pace, slope_ctrl = np.nan, np.nan
                
            by_fight.append((r1_pace, ratio_r3_r1, slope_pace, slope_ctrl,
                             np.nanmean(pace), np.nanmean(ctrl)))
                             
        if not by_fight:
            continue
            
        arr = np.array(by_fight, float)
        
        # Average over all past fights
        rows.append({
            'fighter_id': fid,
            'behav_r1_pace_mean': float(np.nanmean(arr[:,0])),
            'behav_r3_r1_ratio_mean': float(np.nanmean(arr[:,1])),
            'behav_pace_slope_mean': float(np.nanmean(arr[:,2])),
            'behav_ctrl_slope_mean': float(np.nanmean(arr[:,3])),
            'behav_pace_volatility': float(np.nanstd(arr[:,4])),
            'behav_ctrl_volatility': float(np.nanstd(arr[:,5])),
            'behav_n_fights_round': int(arr.shape[0])
        })
        
    return pd.DataFrame(rows)
