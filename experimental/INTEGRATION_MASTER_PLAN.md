# FightIQ: Integration Master Plan & Final Experiment Summary

## Executive Summary
We have completed **15 experimental features** to optimize the FightIQ prediction engine.
The result is a clear roadmap for a **Dual-Mode System**:
1.  **Analyst Mode**: Maximizes **Accuracy (69.9%)** and **Calibration** using Ensembles and Conformal Prediction.
2.  **Gambler Mode**: Maximizes **ROI (42.5%)** using Profit-Aware Loss and Kelly Staking.

## Experiment Summary Table (1-15)

| ID | Feature | Status | Outcome | Verdict |
|:---|:---|:---|:---|:---|
| **1** | **Siamese Net** | ✅ Success | **69.15% Acc**. Excellent ranking. | **Keep** (Ensemble Member) |
| **2** | **PEAR** (Pace) | ✅ Success | **+0.75% Acc**. Good regularization. | **Keep** (Feature) |
| **3** | **TabNet** | ❌ Failed | Complexity > Value. | **Discard** |
| **4** | **Neural ODE** | ❌ Failed | Complexity > Value. | **Discard** |
| **5** | **Deep & Cross** | ❌ Failed | Marginal gains. | **Discard** |
| **6** | **AFM** (Auto-Feat) | ❌ Failed | Added noise. | **Discard** |
| **7** | **Propensity** | ❌ Failed | Degraded Accuracy. | **Discard** |
| **8** | **Stacking** | ✅ Success | **0.5864 Log Loss**. Best Calibration. | **Keep** (Meta-Learner) |
| **9** | **Conformal** | ✅ Success | **85% Coverage**. Quantifies Uncertainty. | **Keep** (Analyst Mode) |
| **10** | **Profit-Loss** | 💰 Profit | **42% ROI**. Beats the market. | **Keep** (Gambler Mode) |
| **11** | **Behavioural** | ❌ Failed | "Pace Slope" added noise. | **Discard** |
| **12** | **Dynamic Elo** | ✅ Success | **+0.56% Acc**. Better ratings. | **Keep** (Replace Std Elo) |
| **13** | **Strength Adj** | ❌ Failed | "Double counting" Elo hurt model. | **Discard** |
| **14** | **Common Opps** | ✅ Success | **+0.32% Acc**. "Win % Diff" is key. | **Keep** (Feature) |
| **15** | **Stoppage** | ✅ Success | **+0.24% Acc**. "Kill/Be Killed" signal. | **Keep** (Feature) |
| **16** | **REA** (Error Analysis) | ✅ Success | **68.26% Acc**. Data depth is key. | **Keep** (Analysis Tool) |
| **17** | **Losing-Learning** | ⏳ Pending | "Film Study" for the model. | **Planned Phase 3** |

## The Winning Architecture

We will integrate the successful components into a production pipeline.

### 1. Feature Engineering Layer
*   **Base**: Physical stats (Reach, Age), Record.
*   **Dynamic Elo** (Exp 12): Replaces standard Elo. K-factor boosts for new fighters & finishes.
*   **PEAR** (Exp 2): Pace and cardio metrics.
*   **Common Opponents** (Exp 14): `common_win_pct_diff`.
*   **Stoppage Propensity** (Exp 15): `finish_rate`, `been_finished_rate`.

### 2. Model Layer (The Hybrid Ensemble)
*   **Member 1**: **XGBoost** (Trained on all features).
*   **Member 2**: **Siamese Network** (Trained on embeddings).
*   **Meta-Learner**: **Time-Anchored Stacking** (Exp 8) to combine them and calibrate probabilities.

### 3. Decision Layer (Dual-Mode)

#### Mode A: "The Analyst" (Truth-Seeking)
*   **Goal**: Know who will win.
*   **Output**: Calibrated Probability (e.g., "Makhachev 72%").
*   **Uncertainty**: **Conformal Prediction** (Exp 9).
    *   *Green*: "Singleton Set" (High Confidence).
    *   *Yellow*: "Both" (Uncertain/Abstain).

#### Mode B: "The Gambler" (Profit-Seeking)
*   **Goal**: Make money.
*   **Model**: **Profit-Weighted XGBoost** (Exp 10).
*   **Strategy**: **Kelly Criterion** (Fractional).
*   **Output**: "Bet 2% of Bankroll on Underdog" (Positive EV only).

## Integration Steps
1.  **Update Feature Pipeline**: Integrate `dynamic_elo.py`, `common_opponents.py`, `early_stoppage.py` into the main `features/` directory.
2.  **Retrain Models**: Train the Hybrid Ensemble and Profit Model on the full dataset with new features.
3.  **Deploy**: Update API to serve both "Analyst" and "Gambler" predictions.
