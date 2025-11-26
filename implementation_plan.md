# Implementation Plan - Phase 3: Retrospective Losing-Learning (REA)

## Goal
Implement a "Film Study" loop where the model learns from its past high-confidence mistakes without full retraining. This involves identifying errors, diagnosing the cause (misleading features), and applying retrospective calibration.

## User Review Required
> [!IMPORTANT]
> This is a new experimental pipeline. It will generate a new dataset (`error_episodes.csv`) and a new calibration artifact (`calibration_adjustments.json`).

## Proposed Changes

### 1. Error Episode Generation (`experimental/losing_learning.py`)
-   **Function**: `generate_error_episodes(model, df)`
-   **Logic**:
    -   Predict on historical data.
    -   Filter for **Wrong Predictions**.
    -   Categorize by Confidence:
        -   **High**: > 60% (Critical Errors)
        -   **Medium**: 50-60% (Borderline)
    -   Save to `experimental/data/error_episodes.csv` with metadata (features, probability, actual outcome).

### 2. Diagnostic Pass (`experimental/losing_learning.py`)
-   **Function**: `diagnose_errors(error_df)`
-   **Logic**:
    -   For each error, calculate **Feature Contribution** (using SHAP or simple difference from mean).
    -   Identify features that strongly favored the *loser* (misleading signals).
    -   Generate "Error Fingerprints" (e.g., "High Reach Advantage + Low Takedown Defense = Upset Risk").

### 3. Retrospective Calibration (`v2/models/analyst_ensemble.py`)
-   **Update**: Modify `predict_proba` to apply adjustments.
-   **Logic**:
    -   Load `calibration_adjustments.json`.
    -   If a new matchup matches an "Error Fingerprint", **penalize confidence**.
    -   Example: If `Reach Diff > 10` and `TD Def < 50%` was a common error pattern, reduce probability toward 0.5.

## Verification Plan
### Automated Tests
-   Run `python experimental/losing_learning.py` to generate episodes and diagnostics.
-   Verify `error_episodes.csv` is created and populated.
-   Test `AnalystEnsemble` with and without calibration to see if it "corrects" past mistakes (or at least reduces confidence in them).
