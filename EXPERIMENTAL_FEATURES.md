# Experimental Features Analysis

## Current Feature Count

Based on `src/data/data_preprocessing.py`, the model uses approximately **50-60 engineered features** (varies based on data availability).

## Feature Categories Breakdown

### 1. Physical Attributes (8 features)
- height_diff, height_ratio
- reach_diff, reach_ratio  
- weight_diff, weight_ratio
- age_diff, age_advantage

### 2. Career Statistics (5 features)
- f_1_total_fights, f_2_total_fights
- f_1_win_pct, f_2_win_pct
- win_pct_diff
- experience_diff, experience_ratio

### 3. Striking Stats (5 features)
- slpm_diff (strikes landed per minute)
- str_acc_diff (strike accuracy)
- sapm_diff (strikes absorbed per minute)
- str_def_diff (strike defense)
- strike_efficiency_diff

### 4. Grappling Stats (6 features)
- td_avg_diff (takedown average)
- td_acc_diff (takedown accuracy)
- td_def_diff (takedown defense)
- sub_avg_diff (submission average)
- grappling_advantage (composite)
- defense_advantage (composite)

### 5. Composite Scores (2 features)
- skill_score_diff
- physical_advantage

### 6. Ranking Features (7 features)
- f_1_ranked, f_2_ranked
- ranked_diff
- ranking_diff, ranking_advantage
- f_1_top5, f_1_top10, f_2_top5, f_2_top10

### 7. Betting Odds (5 features)
- odds_diff
- f_1_favorite
- f_1_implied_prob, f_2_implied_prob
- implied_prob_diff

### 8. Fight Context (2 features)
- title_fight
- num_rounds

### 9. Stance Features (5 features)
- f_1_stance_orthodox, f_1_stance_southpaw
- f_2_stance_orthodox, f_2_stance_southpaw
- stance_match

### 10. Feature Interactions (5 features)
- win_pct_x_odds
- striking_x_accuracy
- experience_x_age_f1, experience_x_age_f2
- reach_x_height
- skill_x_physical

### 11. Polynomial Features (12 features)
- win_pct_diff_squared, win_pct_diff_cubed
- odds_diff_squared, odds_diff_cubed
- slpm_diff_squared, slpm_diff_cubed
- implied_prob_diff_squared, implied_prob_diff_cubed

**Total: ~62 features** (some may be missing if data unavailable)

---

## 🚫 Leaky Features (Excluded)

### Fight Outcomes (~5 features)
- winner, result, result_details, finish_round, finish_time

### Round-by-Round Stats (~50+ features)
- All `f_1_r1_*`, `f_1_r2_*`, `f_1_r3_*`, `f_1_r4_*`, `f_1_r5_*`
- All `f_2_r1_*`, `f_2_r2_*`, `f_2_r3_*`, `f_2_r4_*`, `f_2_r5_*`

### Fight-Specific Stats (~20 features)
- knockdowns, total_strikes_att/succ, sig_strikes_att/succ
- takedown_att/succ, submission_att, reversals, ctrl_time_sec
- (for both fighters)

### Metadata (~15 features)
- Names, URLs, locations, identifiers

**Total Excluded: ~90+ features**

---

## 🎯 Key Insight

The model uses **career averages** (known before fight) but excludes **fight-specific stats** (only known after fight).

This ensures no data leakage - we're only using information a bettor would have before placing a bet!

---

## 🧪 Experimental Feature Ideas

### Ideas to Test

**Feature-Level Ideas (1-5):**
1. **CST – Counterfactual Style Transport** - ✅ COMPLETE: -0.74% accuracy (not helpful) - See `experimental/FEATURE_1_CST_SUMMARY.md`
2. **PEAR – Pace-Elasticity & Attrition Response** - ⏳ IN PROGRESS: Implementation complete, ready for testing
3. **SDA – Strategic Diversity & Adaptability** - Measure entropy and Jensen-Shannon divergence of fight strategy mixes over time
4. **REST – Ref/Commission Stoppage Tolerance prior** - Learn finish rate multipliers by referee/commission/venue combinations
5. **AFM – Adversarial Fragility Margin** - Compute sensitivity margin for adversarial perturbations to matchup function

**Pipeline/Modelling Ideas (6-10):**
6. **Siamese matchup net** - Neural network with symmetry constraint and odds-dropout regularization
7. **Matchmaking propensity weighting** - Weight training examples by inverse propensity of matchmaking
8. **Time-anchored stacking with cross-fitting** - Stacking ensemble with temporal blocks and out-of-fold predictions
9. **Split conformal selective prediction** - Use conformal prediction to abstain on uncertain predictions
10. **Profit-aware loss & staking helpers** - Custom loss functions and Kelly criterion staking utilities

**Round-by-Round Ideas (11-15):**
11. **Behavioural profiles from rounds** - Gas tank, pace curves, front-runner vs late-surger indicators
12. **Round variance & consistency metrics** - Within-round and across-round variability measures
13. **Polynomial fatigue curves** - R1→R3/R5 slopes & curvature from quadratic fits
14. **Opponent-normalised round metrics** - Performance vs opponent archetypes (pressure striker, grappler, etc.)
15. **Sequence-encoder stubs** - RNN/Transformer-based embeddings for round sequences

---

## 📋 Testing Protocol

### Baseline Results
- **Status**: ✅ Complete (Fresh Run - Nov 17, 2025)
- **Best Model**: Stacking Ensemble
- **Test Accuracy**: 0.6209 (62.09%)
- **Test ROC-AUC**: 0.6625
- **Test F1-Score**: 0.7020
- **Date**: Fresh run completed successfully

**Fresh Baseline Models:**
- **Stacking Ensemble**: Accuracy 0.6209, ROC-AUC 0.6625, F1 0.7020 ⭐ BEST
- **XGBoost**: Accuracy 0.6160, ROC-AUC 0.6312, F1 0.6980
- **CatBoost**: Accuracy 0.6160, ROC-AUC 0.6230, F1 0.7004
- **LightGBM**: Accuracy 0.6010, ROC-AUC 0.6174, F1 0.6838

**Configuration:**
- Feature Selection: Enabled (103 features selected)
- Stacking: Enabled
- Optimal Threshold: 0.5933
- Train/Val/Test: 6264/1566/401 fights

### Feature Testing Log
| # | Feature | Status | Accuracy | ROC-AUC | F1-Score | Notes |
|---|---------|--------|----------|---------|----------|-------|
| Baseline | Stacking Ensemble | ✅ Complete | 0.6209 | 0.6625 | 0.7020 | Fresh baseline run (Nov 17, 2025) |
| 1 | CST – Counterfactual Style Transport | ✅ Complete | 0.6135 | - | - | **-0.74%** - Decreased accuracy. 85% coverage but not predictive. See `experimental/FEATURE_1_CST_SUMMARY.md` |
| 2 | PEAR – Pace-Elasticity & Attrition Response | ✅ Complete | 0.6284 | 0.6549 | 0.7084 | **+0.75%** - Improved accuracy! 61.8% F1 coverage, 48.2% F2 coverage. See `experimental/FEATURE_2_PEAR_RESULTS.md` |
| 3 | SDA – Strategic Diversity & Adaptability | ⏳ Pending | - | - | - | - |
| 4 | REST – Ref/Commission Stoppage Tolerance | ⏳ Pending | - | - | - | - |
| 5 | AFM – Adversarial Fragility Margin | ⏳ Pending | - | - | - | - |
| 6 | Siamese matchup net | ⏳ Pending | - | - | - | - |
| 7 | Matchmaking propensity weighting | ⏳ Pending | - | - | - | - |
| 8 | Time-anchored stacking | ⏳ Pending | - | - | - | - |
| 9 | Split conformal selective prediction | ⏳ Pending | - | - | - | - |
| 10 | Profit-aware loss & staking | ⏳ Pending | - | - | - | - |
| 11 | Behavioural profiles from rounds | ⏳ Pending | - | - | - | - |
| 12 | Round variance & consistency | ⏳ Pending | - | - | - | - |
| 13 | Polynomial fatigue curves | ⏳ Pending | - | - | - | - |
| 14 | Opponent-normalised round metrics | ⏳ Pending | - | - | - | - |
| 15 | Sequence-encoder stubs | ⏳ Pending | - | - | - | - |

---

## 📝 Implementation Notes

*Note: The code structure for these 15 ideas is provided in the code block below. This will be extracted into separate modules for testing.*

ROOT = "/mnt/data/fightiq_15ideas_lab"
PKG = os.path.join(ROOT, "fightiq_lab")
FEATURES = os.path.join(PKG, "features")
MODELS = os.path.join(PKG, "models")
DECISION = os.path.join(PKG, "decision")
UTILS = os.path.join(PKG, "utils")
EXAMPLES = os.path.join(ROOT, "examples")

for d in [ROOT, PKG, FEATURES, MODELS, DECISION, UTILS, EXAMPLES]:
    os.makedirs(d, exist_ok=True)

# __init__
with open(os.path.join(PKG, "__init__.py"), "w") as f:
    f.write("# fightiq_lab – lab for 15 ideas (10 original + 5 round-dynamics)\n")

# LICENSE
license_text = f"""MIT License

Copyright (c) {datetime.datetime.now().year}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
with open(os.path.join(ROOT, "LICENSE"), "w") as f:
    f.write(license_text)

# requirements
req = """numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
torch>=2.0
matplotlib>=3.7
"""
with open(os.path.join(ROOT, "requirements.txt"), "w") as f:
    f.write(req)

# README
readme = """# fightiq_15ideas_lab

This lab bundles:

## A. 10 original ideas

### Feature-level (5)
1. **CST – Counterfactual Style Transport** (`features/cst.py`)  
2. **PEAR – Pace-Elasticity & Attrition Response** (`features/pear.py`)  
3. **SDA – Strategic Diversity & Adaptability** (`features/sda.py`)  
4. **REST – Ref/Commission Stoppage Tolerance prior** (`features/rest.py`)  
5. **AFM – Adversarial Fragility Margin** (`features/afm.py`)  

### Pipeline / modelling (5)
6. **Siamese matchup net** with symmetry & odds-dropout (`models/siamese.py`)  
7. **Matchmaking propensity weighting** (`models/propensity_weighting.py`)  
8. **Time-anchored stacking with cross-fitting** (`models/stacking.py`)  
9. **Split conformal selective prediction** (`decision/conformal.py`)  
10. **Profit-aware loss & staking helpers** (`decision/profit_objectives.py`)  

## B. 5 new round-by-round usage ideas (implemented as helpers)

11. **Behavioural profiles from rounds** – gas tank, pace curves, front-runner vs. late-surger (`features/round_dynamics.py`)  
12. **Round variance & consistency metrics** (`features/round_dynamics.py`)  
13. **Polynomial fatigue curves** (R1→R3/R5 slopes & curvature) (`features/round_dynamics.py`)  
14. **Opponent-normalised round metrics** – performance vs archetypes (`features/round_dynamics.py`)  
15. **Sequence-encoder stubs** – signatures for RNN/Transformer-based embeddings (`features/round_dynamics.py`)  

The round modules are deliberately **leak-safe** by design: you pass in historical round-by-round tables **up to a cut-off date**, and they aggregate to pre-fight features.

See `examples/run_round_dynamics_skeleton.py` for how to wire your own tables.
"""
with open(os.path.join(ROOT, "README.md"), "w") as f:
    f.write(readme)

# utils/calibration
calibration_py = """import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < (bins[i+1] if i < n_bins-1 else bins[i+1]+1e-12))
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += m.mean() * abs(acc-conf)
    return float(ece)

def temperature_scale(logits, y_true, max_iter: int = 500, lr: float = 0.01) -> float:
    logits = np.asarray(logits, float)
    y = np.asarray(y_true, float)
    T = 1.0
    def nll(Tv):
        z = logits / max(Tv, 1e-6)
        p = 1/(1+np.exp(-z))
        eps = 1e-12
        return -(y*np.log(p+eps)+(1-y)*np.log(1-p+eps)).mean()
    best_T, best_loss = T, nll(T)
    for _ in range(max_iter):
        eps = 1e-4
        g = (nll(T+eps)-nll(T-eps))/(2*eps)
        T_new = max(1e-3, T - lr*g)
        loss = nll(T_new)
        if loss < best_loss:
            best_T, best_loss = T_new, loss
        if abs(T_new-T) < 1e-6:
            break
        T = T_new
    return float(best_T)
"""
with open(os.path.join(UTILS, "calibration.py"), "w") as f:
    f.write(calibration_py)

# --- FEATURES modules (reuse from previous answer but simplified a bit) ---

cst_py = """import numpy as np
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
    assert 'TARGET' in fighter_past.columns, "Provide TARGET col with one True row as upcoming opponent style."
    target_row = fighter_past.loc[fighter_past['TARGET']].iloc[0]
    target_x = target_row[opp_style_cols].to_numpy(float)
    src = fighter_past.loc[~fighter_past['TARGET']].copy()
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
"""
with open(os.path.join(FEATURES, "cst.py"), "w") as f:
    f.write(cst_py)

pear_py = """import numpy as np
import pandas as pd
from typing import Tuple

def compute_round_efficiency(df_rounds: pd.DataFrame,
                             metrics=('sig_str_diff',),
                             pace_col='opp_sig_str_per_min') -> pd.DataFrame:
    need = {'fight_id','round','fighter_id',metrics[0],pace_col}
    missing = need - set(df_rounds.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df_rounds.copy().sort_values(['fighter_id','fight_id','round'])
    df['lag_eff'] = df.groupby(['fighter_id','fight_id'])[metrics[0]].shift(1)
    df['lag_pace'] = df.groupby(['fighter_id','fight_id'])[pace_col].shift(1)
    df = df.dropna(subset=['lag_eff','lag_pace'])
    df['eff'] = df[metrics[0]].astype(float)
    return df

def fit_pear(df_rounds: pd.DataFrame) -> pd.DataFrame:
    df = compute_round_efficiency(df_rounds)
    rows = []
    for fid, g in df.groupby('fighter_id'):
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
"""
with open(os.path.join(FEATURES, "pear.py"), "w") as f:
    f.write(pear_py)

sda_py = """import numpy as np
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
        mixes = g[mix_cols].to_numpy(float)
        ent = [entropy(m) for m in mixes]
        js = [js_div(mixes[i-1], mixes[i]) for i in range(1,len(mixes))] if len(mixes)>1 else [np.nan]
        rows.append({'fighter_id': fid,
                     'SDA_entropy_mean': float(np.nanmean(ent)),
                     'SDA_js_median': float(np.nanmedian(js)),
                     'n_fights': int(len(g))})
    return pd.DataFrame(rows)
"""
with open(os.path.join(FEATURES, "sda.py"), "w") as f:
    f.write(sda_py)

rest_py = """import numpy as np
import pandas as pd

def fit_rest_priors(fights: pd.DataFrame,
                    group_cols=('referee','commission','venue'),
                    strat_cols=('weight_class','max_rounds')) -> pd.DataFrame:
    need = set(group_cols)|set(strat_cols)|{'finished','outcome_method'}
    missing = need - set(fights.columns)
    if missing:
        raise ValueError(f"Missing: {missing}")
    df = fights.copy()
    base = df.groupby(list(strat_cols))['finished'].mean().rename('base_finish').reset_index()
    grp = df.groupby(list(group_cols)+list(strat_cols))['finished'].mean().rename('grp_finish').reset_index()
    out = grp.merge(base, on=list(strat_cols), how='left')
    out['REST_finish_mult'] = (out['grp_finish']+1e-6)/(out['base_finish']+1e-6)
    return out

def apply_rest_prior(row: pd.Series, rest_table: pd.DataFrame) -> float:
    strat_cols = [c for c in ('weight_class','max_rounds') if c in rest_table.columns]
    for keys in [('referee','commission','venue'),('commission','venue'),('venue',)]:
        cols = [c for c in keys if c in rest_table.columns]
        if not cols:
            continue
        m = rest_table
        for c in cols:
            m = m[m[c]==row.get(c, None)]
        for sc in strat_cols:
            m = m[m[sc]==row.get(sc, None)]
        if len(m)>0:
            return float(m['REST_finish_mult'].median())
    return 1.0
"""
with open(os.path.join(FEATURES, "rest.py"), "w") as f:
    f.write(rest_py)

afm_py = """import numpy as np
from typing import Callable

def compute_afm_numeric(f: Callable[[np.ndarray,np.ndarray], float],
                        a: np.ndarray, b: np.ndarray,
                        margin: float = 0.2, eps: float = 1e-4) -> float:
    x = np.concatenate([a,b]).astype(float)
    def fwrap(xv):
        d = len(a)
        return f(xv[:d], xv[d:])
    g = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy(); x1[i]+=eps
        x2 = x.copy(); x2[i]-=eps
        g[i] = (fwrap(x1)-fwrap(x2))/(2*eps)
    gnorm = np.linalg.norm(g)+1e-12
    return float(abs(margin)/gnorm)
"""
with open(os.path.join(FEATURES, "afm.py"), "w") as f:
    f.write(afm_py)

# --- New round_dynamics module implementing 5 extra ideas ---
round_dyn_py = """\"\"\"Round dynamics feature engineering helpers.

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
- archetype_opp (optional, for opponent-normalised stats)
\"\"\"
import numpy as np
import pandas as pd
from typing import List, Dict

def _poly_fit_rounds(rounds: np.ndarray, values: np.ndarray, deg: int = 2):
    \"\"\"Fit y ~ poly(round) of given degree. Returns coefficients (lowest degree first).\"\"\"
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
    \"\"\"Compute fighter-level behaviour from historical rounds:
    - avg R1 pace, R3/R1 ratio
    - avg control slope across rounds
    - variability of pace & control.

    Returns one row per fighter summarising **all rows in round_df**.
    For pre-fight usage, call this on a table filtered to fights strictly before the prediction date.
    \"\"\"
    df = round_df.copy()
    need = {fighter_id_col,fight_id_col,date_col,round_col,pace_col,ctrl_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    rows = []
    for fid, g in df.groupby(fighter_id_col):
        # per-fight aggregates
        by_fight = []
        for _, gf in g.groupby(fight_id_col):
            gf = gf.sort_values(round_col)
            r = gf[round_col].to_numpy()
            pace = gf[pace_col].to_numpy(float)
            ctrl = gf[ctrl_col].to_numpy(float)
            r1_pace = pace[r==1].mean() if (r==1).any() else np.nan
            r3_pace = pace[r==3].mean() if (r==3).any() else np.nan
            ratio_r3_r1 = r3_pace / r1_pace if r1_pace and not np.isnan(r1_pace) else np.nan
            # simple linear slope pace ~ round
            if len(r) >= 2:
                A = np.column_stack([np.ones(len(r)), r])
                beta, *_ = np.linalg.lstsq(A, pace, rcond=None)
                slope_pace = beta[1]
                A2 = np.column_stack([np.ones(len(r)), r])
                beta2, *_ = np.linalg.lstsq(A2, ctrl, rcond=None)
                slope_ctrl = beta2[1]
            else:
                slope_pace, slope_ctrl = np.nan, np.nan
            by_fight.append((r1_pace, ratio_r3_r1, slope_pace, slope_ctrl,
                             pace.mean(), ctrl.mean()))
        if not by_fight:
            continue
        arr = np.array(by_fight, float)
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

def build_fatigue_polynomials(round_df: pd.DataFrame,
                              fighter_id_col: str = 'fighter_id',
                              fight_id_col: str = 'fight_id',
                              round_col: str = 'round',
                              pace_col: str = 'pace') -> pd.DataFrame:
    \"\"\"Fit quadratic polynomial per fight, then pool coefficients per fighter.

    y(r) ~ a + b*r + c*r^2

    Returns fighter-level mean coeffs a,b,c (and their std) as fatigue-shape features.
    \"\"\"
    df = round_df.copy()
    need = {fighter_id_col,fight_id_col,round_col,pace_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    rows = []
    for fid, g in df.groupby(fighter_id_col):
        coefs = []
        for _, gf in g.groupby(fight_id_col):
            r = gf[round_col].to_numpy(float)
            pace = gf[pace_col].to_numpy(float)
            c = _poly_fit_rounds(r, pace, deg=2)
            coefs.append(c)
        if not coefs:
            continue
        C = np.vstack(coefs)  # (n_fights, 3)
        rows.append({
            'fighter_id': fid,
            'fatigue_a_mean': float(np.nanmean(C[:,0])),
            'fatigue_b_mean': float(np.nanmean(C[:,1])),
            'fatigue_c_mean': float(np.nanmean(C[:,2])),
            'fatigue_a_std': float(np.nanstd(C[:,0])),
            'fatigue_b_std': float(np.nanstd(C[:,1])),
            'fatigue_c_std': float(np.nanstd(C[:,2])),
            'fatigue_n_fights': int(C.shape[0])
        })
    return pd.DataFrame(rows)

def build_round_variability(round_df: pd.DataFrame,
                            fighter_id_col: str = 'fighter_id',
                            round_col: str = 'round',
                            pace_col: str = 'pace',
                            acc_col: str = 'accuracy') -> pd.DataFrame:
    \"\"\"Compute per-fighter variability within and across rounds for pace & accuracy.\"\"\"
    df = round_df.copy()
    need = {fighter_id_col,round_col,pace_col,acc_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    rows = []
    for fid, g in df.groupby(fighter_id_col):
        rows.append({
            'fighter_id': fid,
            'var_pace_over_rounds': float(g.groupby(round_col)[pace_col].mean().var()),
            'var_acc_over_rounds': float(g.groupby(round_col)[acc_col].mean().var()),
            'var_pace_within': float(g[pace_col].var()),
            'var_acc_within': float(g[acc_col].var()),
            'n_round_rows': int(len(g))
        })
    return pd.DataFrame(rows)

def build_opponent_normalised(round_df: pd.DataFrame,
                              fighter_id_col: str = 'fighter_id',
                              archetype_col: str = 'archetype_opp',
                              pace_col: str = 'pace',
                              ctrl_col: str = 'ctrl_share') -> pd.DataFrame:
    \"\"\"Aggregate per fighter x opponent archetype round metrics.

    E.g. avg pace and control vs 'pressure_striker', 'grappler', etc.
    \"\"\"
    df = round_df.copy()
    need = {fighter_id_col, archetype_col, pace_col, ctrl_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    rows = []
    for (fid, arch), g in df.groupby([fighter_id_col, archetype_col]):
        rows.append({
            'fighter_id': fid,
            'opp_arch': arch,
            'onorm_pace_mean': float(g[pace_col].mean()),
            'onorm_ctrl_mean': float(g[ctrl_col].mean()),
            'onorm_n_rounds': int(len(g))
        })
    return pd.DataFrame(rows)

# sequence encoder stubs are left as signatures since implementation depends on your stack
def sequence_encoder_signature():
    \"\"\"Pseudocode / docstring placeholder for RNN/Transformer-based round encoders.

    Suggested interface (PyTorch):

    class RoundSequenceEncoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_layers):
            ...
        def forward(self, seq, lengths):
            # seq: (batch, max_rounds, in_dim)
            # returns (batch, emb_dim) fighter-level embedding
            ...

    You would train this on past fights only to predict outcome / finish / scores,
    then freeze and use the embeddings as inputs to the matchup model.
    \"\"\"
    return None
"""
with open(os.path.join(FEATURES, "round_dynamics.py"), "w") as f:
    f.write(round_dyn_py)

# --- MODELS ---
siamese_py = """import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseMatchupNet(nn.Module):
    def __init__(self, in_dim: int, ctx_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        pair_in = 2*hidden + hidden + (ctx_dim if ctx_dim else 0)
        self.head = nn.Sequential(
            nn.Linear(pair_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def encode(self, x):
        return self.enc(x)

    def pair_features(self, ea, eb, ctx=None):
        feats = [torch.abs(ea-eb), ea*eb, ea, eb]
        if ctx is not None:
            feats.append(ctx)
        return torch.cat(feats, dim=-1)

    def forward(self, a, b, ctx=None):
        ea, eb = self.encode(a), self.encode(b)
        h = self.pair_features(ea, eb, ctx)
        logit = self.head(h).squeeze(-1)
        return torch.sigmoid(logit)

def symmetric_loss(model, a, b, y, ctx=None, lam_sym: float = 1.0):
    p_ab = model(a,b,ctx)
    p_ba = model(b,a,ctx)
    bce = F.binary_cross_entropy(p_ab, y)
    sym = ((p_ab+p_ba-1.0)**2).mean()
    return bce + lam_sym*sym

class OddsDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    def forward(self, ctx, train: bool = True):
        if (not train) or ctx is None or self.p<=0:
            return ctx
        mask = (torch.rand_like(ctx) > self.p).float()
        return ctx*mask
"""
with open(os.path.join(MODELS, "siamese.py"), "w") as f:
    f.write(siamese_py)

propensity_py = """import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def fit_matchmaking_propensity(pairs_df: pd.DataFrame,
                               feature_cols,
                               label_col: str = 'made_match') -> LogisticRegression:
    X = pairs_df[feature_cols].to_numpy(float)
    y = pairs_df[label_col].to_numpy(int)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(X, y)
    return clf

def importance_weights(clf: LogisticRegression, X: np.ndarray, w_max: float = 10.0) -> np.ndarray:
    q = clf.predict_proba(X)[:,1]
    eps = 1e-6
    w = 1.0/np.clip(q, eps, None)
    return np.clip(w, 1.0, w_max)
"""
with open(os.path.join(MODELS, "propensity_weighting.py"), "w") as f:
    f.write(propensity_py)

stacking_py = """import numpy as np
import pandas as pd
from typing import Dict, Callable, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def expanding_blocks(dates: pd.Series, n_splits: int = 5) -> List[Tuple[np.ndarray,np.ndarray]]:
    uniq = np.array(sorted(dates.unique()))
    if len(uniq) < 2:
        return []
    splits = np.array_split(uniq[1:], n_splits)
    blocks = []
    for test_dates in splits:
        if len(test_dates) == 0: 
            continue
        first_test = test_dates[0]
        train_dates = uniq[uniq < first_test]
        tr_idx = dates.isin(train_dates).to_numpy().nonzero()[0]
        te_idx = dates.isin(test_dates).to_numpy().nonzero()[0]
        if len(tr_idx)==0 or len(te_idx)==0:
            continue
        blocks.append((tr_idx, te_idx))
    return blocks

def generate_oof_preds(base_models: Dict[str, Callable],
                       X: np.ndarray, y: np.ndarray, dates: pd.Series):
    oof = {k: np.full(len(y), np.nan) for k in base_models}
    for tr, te in expanding_blocks(dates):
        for name, fit_pred in base_models.items():
            oof[name][te] = fit_pred(X[tr], y[tr], X[te])
    oof_df = pd.DataFrame(oof)
    mask = ~oof_df.isna().any(1)
    return oof_df[mask], y[mask], mask

def fit_meta_learner(oof_df: pd.DataFrame, y: np.ndarray, calibrate: bool = True):
    meta = LogisticRegression(max_iter=1000).fit(oof_df.to_numpy(), y.astype(int))
    iso = None
    if calibrate:
        p = meta.predict_proba(oof_df)[:,1]
        iso = IsotonicRegression(out_of_bounds='clip').fit(p, y.astype(int))
    return meta, iso

def meta_predict(meta, iso, base_preds_df: pd.DataFrame) -> np.ndarray:
    p = meta.predict_proba(base_preds_df)[:,1]
    if iso is not None:
        p = iso.transform(p)
    return p
"""
with open(os.path.join(MODELS, "stacking.py"), "w") as f:
    f.write(stacking_py)

# --- DECISION ---
conformal_py = """import numpy as np

def fit_conformal(y_prob, y_true, alpha: float = 0.1):
    y_prob = np.asarray(y_prob, float)
    y_true = np.asarray(y_true, int)
    score = -(y_true*np.log(y_prob+1e-12) + (1-y_true)*np.log(1-y_prob+1e-12))
    q = np.quantile(score, 1-alpha)
    return float(q)

def predict_set(p: float, q: float) -> str:
    loss1 = -np.log(p+1e-12)
    loss0 = -np.log(1-p+1e-12)
    Aok = loss1 <= q
    Bok = loss0 <= q
    if Aok and not Bok:
        return 'A'
    if Bok and not Aok:
        return 'B'
    return 'AB'
"""
with open(os.path.join(DECISION, "conformal.py"), "w") as f:
    f.write(conformal_py)

profit_py = """import numpy as np

def weighted_log_loss(y_true, y_prob, sample_weight=None):
    y = np.asarray(y_true, float)
    p = np.asarray(y_prob, float)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, float)
    eps = 1e-12
    return float(-(w*(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))).sum()/(w.sum()+1e-12))

def ev_weight(y_prob, p_mkt):
    y_prob = np.asarray(y_prob, float)
    p_mkt = np.asarray(p_mkt, float)
    def logit(x):
        x = np.clip(x, 1e-6, 1-1e-6)
        return np.log(x/(1-x))
    return np.abs(logit(y_prob)-logit(p_mkt))

def kelly_fraction(p: float, dec_odds: float, frac: float = 0.5) -> float:
    b = dec_odds - 1.0
    edge = p*b - (1-p)
    f = edge/b if b>0 else 0.0
    return max(0.0, frac*f)

def card_portfolio_weights(stakes, corr=None, risk_aversion: float = 0.0):
    s = np.asarray(stakes, float)
    if corr is None:
        return s/(s.sum()+1e-12)
    n = len(s)
    I = np.eye(n)
    Sigma = np.asarray(corr, float)
    A = I + risk_aversion*Sigma
    x = np.linalg.solve(A, s)
    x = np.clip(x, 0, None)
    return x/(x.sum()+1e-12)
"""
with open(os.path.join(DECISION, "profit_objectives.py"), "w") as f:
    f.write(profit_py)

# --- Examples ---
example_rounds = """\"\"\"Skeleton to show how to call round_dynamics helpers.

Replace the dummy data with your real round-by-round table.
\"\"\"
import pandas as pd
import numpy as np

from fightiq_lab.features.round_dynamics import (
    build_behavioural_profiles,
    build_fatigue_polynomials,
    build_round_variability,
    build_opponent_normalised,
)

def main():
    rng = np.random.default_rng(0)
    df_rounds = pd.DataFrame({
        'fighter_id': np.repeat(['A','B'], 9),
        'opponent_id': np.repeat(['X','Y'], 9),
        'fight_id': np.repeat(np.arange(3), 6),
        'fight_date': pd.date_range('2020-01-01', periods=18, freq='15D'),
        'round': list(np.tile([1,2,3], 6)),
        'pace': rng.normal(4.0, 0.5, size=18),
        'accuracy': np.clip(rng.normal(0.45, 0.05, size=18), 0, 1),
        'ctrl_share': np.clip(rng.normal(0.4, 0.2, size=18), 0, 1),
        'archetype_opp': np.repeat(['pressure','grappler','pressure'], 6),
    })

    print(\"Behavioural profiles:\\n\", build_behavioural_profiles(df_rounds))
    print(\"Fatigue polynomials:\\n\", build_fatigue_polynomials(df_rounds))
    print(\"Round variability:\\n\", build_round_variability(df_rounds))
    print(\"Opponent-normalised:\\n\", build_opponent_normalised(df_rounds))

if __name__ == '__main__':
    main()
"""
with open(os.path.join(EXAMPLES, "run_round_dynamics_skeleton.py"), "w") as f:
    f.write(example_rounds)

# Zip it
zip_path = "/mnt/data/fightiq_15ideas_lab.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for folder, _, files in os.walk(ROOT):
        for file in files:
            fp = os.path.join(folder, file)
            z.write(fp, arcname=os.path.relpath(fp, ROOT))

zip_path
