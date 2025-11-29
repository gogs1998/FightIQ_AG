# FightIQ Future Roadmap

## 🧪 The Lab: 10 Experimental Ideas (Folder: `experiment_2`)

These are high-risk, high-reward concepts to push the boundaries of sports prediction.

1.  **The "Trash Talk" Sentiment Engine (NLP)**
    *   **Concept:** Scrape pre-fight interviews, press conferences, and social media. Use an LLM to analyze "Confidence" vs. "Anxiety" markers.
    *   **Hypothesis:** A sudden drop in confidence or over-aggression in speech patterns correlates with poor performance.

2.  **Gym/Camp Elo Ratings**
    *   **Concept:** Instead of just Fighter Elo, track the Elo of the *Gym* (e.g., American Top Team, City Kickboxing).
    *   **Hypothesis:** Fighters moving to a high-Elo gym get a "buff" that the market underestimates for their first fight.

3.  **The "MMA Math" Graph Neural Network (GNN)**
    *   **Concept:** Represent the entire UFC history as a graph where nodes are fighters and edges are fight results. Use a GNN (GraphSAGE or GAT) to learn embeddings.
    *   **Hypothesis:** GNNs can capture transitive properties ("A beat B, B beat C, so A beats C") better than tabular models.

4.  **"Chin Health" Decay Model**
    *   **Concept:** Create a specific feature tracking "Cumulative Head Strikes Absorbed" and "Knockdowns" over a career, with an exponential decay factor.
    *   **Hypothesis:** Predict sudden "chin failure" (KO losses) for aging veterans before the market sees it.

5.  **Judge Bias Detector**
    *   **Concept:** For fights likely to go to decision, model the specific judges assigned (if known) or general judging trends in that location/commission.
    *   **Hypothesis:** Some judges favor wrestling control; others favor damage. Matching fighter style to judge tendencies = edge.

6.  **The "Steam Chaser" (Market Dynamics)**
    *   **Concept:** Don't predict the fight; predict the *odds movement*. Train a time-series model on odds history (opening vs. closing).
    *   **Hypothesis:** Smart money moves early. If we can predict where the line will close, we can arbitrage or beat CLV consistently.

7.  **Biometric/Physique Analysis (Computer Vision)**
    *   **Concept:** Use CV to analyze weigh-in photos. Detect "soft" bodies vs. "shredded" conditioning or bad weight cuts (sunken eyes).
    *   **Hypothesis:** Visual cues of a bad weight cut are a massive signal that data misses.

8.  **Referee Tendency Model**
    *   **Concept:** Some refs stop fights early (Herb Dean?); others let them die (Mario Yamasaki style).
    *   **Hypothesis:** Affects "Over/Under" and "Method" bets. A strict ref increases TKO probability.

9.  **Genetic Algorithm Strategy Optimizer**
    *   **Concept:** Instead of fixed Kelly staking, let a genetic algorithm evolve thousands of betting strategies (stake size, confidence thresholds, parlay logic) over historical data.
    *   **Hypothesis:** The optimal strategy is likely dynamic and non-linear.

10. **Live Betting "Momentum" Bot**
    *   **Concept:** Ingest live fight stats (Round 1 strikes) via API during the event. Predict the winner of Round 2/3 live.
    *   **Hypothesis:** Markets overreact to a flashy Round 1. Real-time data is the ultimate edge.

---

## 🎯 The "Prop Hunter": Round & Method Model Plan

**Objective:** Predict **HOW** (KO, Sub, Dec) and **WHEN** (R1, R2, R3) a fight ends.
**Target Markets:** "Method of Victory", "Round Betting", "Fight to Go the Distance".

### 1. Data Engineering (The "Finish" Features)
We need new features specifically for finishes:
*   **Finish Rate:** `(Wins by KO + Wins by Sub) / Total Wins`
*   **Survival Rate:** `(Losses by Dec) / Total Losses`
*   **Avg Fight Time:** Does he finish early or grind to a decision?
*   **"Gassed" Factor:** Strike output in R3 vs. R1 (Cardio proxy).
*   **Sub Offense vs. Sub Defense:** Differential is critical for submission props.

### 2. The Architecture: Hierarchical Ensemble
Instead of one giant model, we build a "Decision Tree" of models:

*   **Model A: The "Distance" Classifier (Binary)**
    *   *Question:* Will this fight go to decision? (Yes/No)
    *   *Target:* `is_decision`
    *   *Use Case:* Betting "Fight Goes the Distance" (FGTD).

*   **Model B: The "Method" Classifier (Multi-class)**
    *   *Question:* If it ends inside the distance, how?
    *   *Target:* `[KO/TKO, Submission]`
    *   *Input:* Only fights that *didn't* go to decision.

*   **Model C: The "Time" Regressor (or Multi-class)**
    *   *Question:* If it finishes, in which round?
    *   *Target:* `[Round 1, Round 2, Round 3, Round 4, Round 5]`
    *   *Input:* Only finished fights.

### 3. The Probability Tree
To get the probability of **"Fighter A by KO in Round 1"**:

$$ P(A_{Win}) \times P(Finish | A_{Win}) \times P(KO | Finish) \times P(R1 | KO) $$

### 4. Implementation Steps
1.  **Refine Target Labels:** Parse `training_data.csv` to create `win_method` (KO, SUB, DEC) and `end_round` columns.
2.  **Train Model A (FGTD):** XGBoost Classifier.
3.  **Train Model B (Method):** XGBoost Classifier (Multi-class).
4.  **Train Model C (Round):** XGBoost Classifier (Multi-class) or Poisson Regression.
5.  **Combine:** Create a script `predict_props.py` that outputs the full probability matrix.

### 5. Verification
*   **Metric:** Log Loss is king here. Accuracy is misleading for rare events (like Draw or R5 Sub).
*   **Calibration:** Crucial. If we say "R1 KO is 20%", it must happen 20% of the time. Use Isotonic Regression on each leaf of the tree.
