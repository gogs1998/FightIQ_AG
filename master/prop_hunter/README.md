# 🏹 Prop Hunter: Method & Round Prediction Engine

**Objective:** Predict *how* and *when* a fight will end to exploit high-yield Prop Bets (e.g., "Taira by Sub Round 1").

## 🧠 The Strategy: Hierarchical Modeling
Instead of one giant model, we build a tree of specialized "Lean" models using the 51 Boruta-confirmed features.

### 1. The "Finish" Model (Binary)
*   **Question:** Will this fight end inside the distance?
*   **Target:** `1` (Finish), `0` (Decision).
*   **Why:** This is the foundational filter. If `P(Finish) < 30%`, we don't bet props.

### 2. The "Method" Model (Multi-Class)
*   **Question:** If it finishes, how?
*   **Target:** `KO/TKO`, `Submission`. (Decision is handled by Model 1).
*   **Logic:** Trained *only* on fights that finished.

### 3. The "Round" Model (Regression/Multi-Class)
*   **Question:** When will it end?
*   **Target:** `Round 1`, `Round 2`, `Round 3`, `Round 4`, `Round 5`.
*   **Logic:** Trained on all fights (Decision = Max Rounds).

## 🛠️ Tech Stack
*   **Features:** Only the 51 "All-Star" features from `features.json` (Boruta).
*   **Algorithm:** XGBoost (Optimized for Multi-Class LogLoss).
*   **Calibration:** Isotonic Calibration for each class probability.

## 🚀 Workflow
1.  `train_finish_model.py`: Train the "Does it go the distance?" model.
2.  `train_method_model.py`: Train the "KO vs Sub" model.
3.  `predict_props.py`: Combine them to output:
    *   `P(Win & KO)`
    *   `P(Win & Sub)`
    *   `P(Win & Dec)`
