# Implementation Plan: Master Props (Prop Hunter 2.0)

## Goal
Build a specialized pipeline for **Prop Betting** (Method & Round) that leverages the superior feature engineering of `master_3` to achieve >70% precision on high-confidence KO/TKO predictions and profitable ROI on Submission and Round props.

## User Review Required
> [!IMPORTANT]
> We are migrating from the old `master/prop_hunter` to a new `master_props` directory. We will reuse the `master_3` feature set (`features_selected.json`) as the foundation, adding prop-specific features only if necessary.

## Proposed Changes

### 1. Feature Engineering
*   **Base**: Use `master_3/features_selected.json` (proven high signal).
*   **Additions**:
    *   `ko_ratio`: KO Wins / Total Wins.
    *   `sub_ratio`: Sub Wins / Total Wins.
    *   `chin_score`: From `master_3/features/chin.py`.
    *   `finish_rate`: (KO + Sub) / Total Fights.

### 2. Modeling Architecture (Hierarchical)
We will train 3 distinct models:
1.  **Finish Model (Binary)**: `GTD` (Decision) vs `ITD` (Finish).
    *   *Target*: `is_finish` (0/1).
2.  **Method Model (Multi-Class)**: `KO/TKO` vs `Submission` (Conditional on Finish).
    *   *Target*: `method` (0=KO, 1=Sub). Trained only on finish rows.
3.  **Round Model (Regression/Class)**: Exact Round (1-5).
    *   *Target*: `round_num`.

### 3. Training Pipeline
*   **Script**: `master_props/train_props.py`.
*   **Logic**:
    *   Load `master_3` data.
    *   Train XGBoost classifiers for Finish and Method.
    *   Train XGBoost regressor (or classifier) for Round.
    *   Calibrate probabilities using Isotonic Regression.

### 4. Betting Strategy ("The Green Light")
Implement the "Sniper" logic:
*   **KO Bet**: `P(Win) * P(Finish) * P(KO|Finish) > Threshold` (e.g., 50%).
*   **Sub Bet**: `P(Win) * P(Finish) * P(Sub|Finish) > Threshold` (e.g., 30%).
*   **Round Bet**: `P(Round X) > Threshold`.

## Verification Plan

### Automated Tests
*   **Precision Check**: Run `evaluate_prop_precision.py` on 2024-2025 holdout.
*   **Target**:
    *   KO Precision > 60% (at >2.0 odds).
    *   Sub ROI > 10%.

### Manual Verification
*   **Betting Log**: Generate `props_log.csv` for the holdout set to verify ROI.
