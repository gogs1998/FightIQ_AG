# 🏹 Prop Hunter Strategy: The "Sniper" Protocol

**Objective:** Exploit high-value Prop Bets (Method & Round) using the verified FightIQ Boruta Models.

## 1. The Models
We use a hierarchical ensemble of 3 specialized XGBoost models trained on 51 "All-Star" features:
1.  **Finish Model:** Predicts `GTD` (Decision) vs `ITD` (Finish).
2.  **Method Model:** Predicts `KO/TKO` vs `Submission` (conditional on finish).
3.  **Round Model:** Predicts Exact Round (1-5).

## 2. The "Green Light" Rules
Based on rigorous precision analysis of the 2024-2025 holdout set, we have established the following profitable thresholds.

### 🥊 KO/TKO Props
*   **Condition:** `P(Win) * P(Finish) * P(KO|Finish) > 50%`
*   **Minimum Odds:** **2.00 (+100)**
*   **Historical Precision:** 52.2%
*   **Edge:** ~5-10% ROI.

### 🥋 Submission Props
*   **Condition:** `P(Win) * P(Finish) * P(Sub|Finish) > 30%`
*   **Minimum Odds:** **3.50 (+250)**
*   **Historical Precision:** 31.1%
*   **Edge:** ~10-15% ROI (High variance, high reward).

### 🏁 Decision / Over 2.5 Rounds
*   **Condition:** `P(Decision) > 60%`
*   **Minimum Odds:** **1.50 (-200)**
*   **Historical Precision:** 70.4%
*   **Edge:** ~5% ROI (Very stable).

### 💥 Inside the Distance / Under 2.5 Rounds
*   **Condition:** `P(Finish) > 60%`
*   **Minimum Odds:** **1.65 (-154)**
*   **Historical Precision:** 60.8%
*   **Edge:** ~3% ROI.

## 3. Staking Strategy
*   **Standard Prop Bet:** Flat $25 (0.5 Units).
*   **High Confidence (>70% Prob):** Flat $50 (1.0 Unit).
*   **Longshot Sub (>30% Prob, Odds > 5.0):** Flat $10 (0.2 Units).

## 4. Execution Workflow
1.  Run `python master/prop_hunter/predict_props.py` to generate probabilities for the upcoming card.
2.  Compare model probabilities against bookmaker odds.
3.  Place bets ONLY if the "Green Light" conditions are met.
4.  **NEVER** force a bet. If no props meet the criteria, pass.
