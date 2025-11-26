import numpy as np
import pandas as pd

def fit_rest_priors(fights: pd.DataFrame,
                    group_cols=('referee','event_state','event_country'),
                    strat_cols=('weight_class','max_rounds')) -> pd.DataFrame:
    """
    Calculate finish rate multipliers for groups (referee, location).
    """
    need = set(group_cols)|set(strat_cols)|{'finished'}
    missing = need - set(fights.columns)
    # if missing:
    #     raise ValueError(f"Missing: {missing}")
    
    # Handle missing columns gracefully by ignoring them in groups
    actual_groups = [c for c in group_cols if c in fights.columns]
    actual_strats = [c for c in strat_cols if c in fights.columns]
    
    if not actual_groups:
        return pd.DataFrame()
        
    df = fights.copy()
    
    # Base finish rate per stratum (e.g. weight class)
    if actual_strats:
        base = df.groupby(actual_strats)['finished'].mean().rename('base_finish').reset_index()
        # Group finish rate
        grp = df.groupby(actual_groups + actual_strats)['finished'].mean().rename('grp_finish').reset_index()
        out = grp.merge(base, on=actual_strats, how='left')
    else:
        base_val = df['finished'].mean()
        grp = df.groupby(actual_groups)['finished'].mean().rename('grp_finish').reset_index()
        out = grp.copy()
        out['base_finish'] = base_val
        
    # Calculate multiplier: Group Finish Rate / Base Finish Rate
    # e.g. If Herb Dean finishes 60% of HW fights, and avg HW finish is 50%, mult = 1.2
    out['REST_finish_mult'] = (out['grp_finish']+1e-6)/(out['base_finish']+1e-6)
    
    return out

def apply_rest_prior(row: pd.Series, rest_table: pd.DataFrame) -> float:
    """
    Apply the most specific prior available for a row.
    """
    if rest_table.empty:
        return 1.0
        
    # We need to match row values to rest_table
    # rest_table has columns: [group_cols] + [strat_cols] + REST_finish_mult
    
    # Filter rest_table to match row's stratum
    # strat_cols = [c for c in ('weight_class','max_rounds') if c in rest_table.columns]
    # But we don't know which ones were used.
    # Let's assume rest_table columns excluding 'grp_finish','base_finish','REST_finish_mult' are the keys.
    
    keys = [c for c in rest_table.columns if c not in ['grp_finish','base_finish','REST_finish_mult']]
    
    # This is slow if we do it row by row with filtering.
    # Better to use merge in the main pipeline.
    # This function is for single-row inference if needed.
    
    m = rest_table
    for k in keys:
        val = row.get(k)
        if pd.isna(val): continue
        if k in m.columns:
            m = m[m[k] == val]
            
    if len(m) > 0:
        # If multiple matches (shouldn't happen if keys are unique), take mean
        return float(m['REST_finish_mult'].mean())
        
    return 1.0
