# 🗺️ FightIQ V2 Roadmap: The "Quantopian for MMA" Platform
**Vision:** Move beyond a simple "Prediction Service" to a full-stack **Algorithmic Trading Platform for MMA**. We empower users to build, test, and deploy their own betting strategies using our leak-free data and infrastructure.

---

## 🚀 Phase 1: The "Octagon Terminal" (Visualization Layer)
**Goal:** A "Bloomberg Terminal" for Fight Fans. Shiny, customizable, and data-dense.

### 1.1 The Widget System
*   **Concept:** Users can drag-and-drop "Widgets" onto their dashboard.
*   **Available Widgets:**
    *   **Elo Ticker:** Live Elo ratings for any fighter.
    *   **Tale of the Tape 2.0:** Holographic comparison of Reach, Age, Stance.
    *   **Chin Health Monitor:** Visual bar chart of accumulated damage.
    *   **Steam Tracker:** Live odds movement graph.
    *   **"The Gap":** Visualizing the difference between Implied Probability and True Probability.

### 1.2 "DYOR" Mode (Do Your Own Research)
*   **Interactive Graphs:** Users can plot "Win Rate vs Age Gap" or "Finish Rate vs Weight Class" dynamically.
*   **Fighter Deep Dive:** Click a fighter to see their entire "Career Equity Curve" (Betting profitability over time).

---

## 🧪 Phase 2: The "Strategy Lab" (No-Code ML Pipeline)
**Goal:** Let users play Data Scientist. They build the model; we provide the infrastructure.

### 2.1 Feature Selector
*   Users choose their "Alpha Factors" from our verified, leak-free library:
    *   [x] Age Difference
    *   [ ] Reach Advantage
    *   [x] Dynamic Elo
    *   [ ] Gym Quality (New)
    *   [ ] Twitter Sentiment (New)

### 2.2 Model Architect
*   Users select the engine:
    *   **"The Hammer"** (Logistic Regression - Simple, Interpretable).
    *   **"The Scalpel"** (XGBoost - High Precision).
    *   **"The Brain"** (Neural Network - Experimental).

### 2.3 Risk Settings
*   Users define their staking strategy:
    *   Flat Bet ($100/fight).
    *   Kelly Criterion (Aggressive).
    *   "Value Sniper" (Only bet if Edge > X%).

---

## ⚔️ Phase 3: The "Backtest Arena" (Gamification)
**Goal:** "Can you beat FightIQ?"

### 3.1 The Simulation Engine
*   User clicks "Run Backtest".
*   System replays history (2010-2025) using *only* the data available at the time (Strict Leakage Prevention).
*   **Output:** A verified Equity Curve, Sharpe Ratio, and Max Drawdown for *their* strategy.

### 3.2 The Leaderboard
*   Rank user strategies by Risk-Adjusted Return (Sortino Ratio).
*   **"Copy Trading":** Allow users to subscribe to the top-performing user strategies (Revenue Share model).

---

## 🛠️ Technical Architecture (V2)
*   **Frontend:** Next.js + React Grid Layout (for the dashboard).
*   **Backend:** FastAPI (Python) to run the ML pipelines.
*   **Worker Queue:** Celery/Redis to handle the heavy backtesting jobs asynchronously.
*   **Database:** PostgreSQL (User strategies) + ClickHouse (High-speed tick data for odds).

---

## 📅 Execution Plan
1.  **Design:** Wireframe the "Octagon Terminal" (Next.js).
2.  **API:** Expose our `backtest_strategies.py` logic via API endpoints.
3.  **Data:** Ensure all 51 Boruta features are available as selectable inputs.
4.  **Launch:** Release "FightIQ Labs" (Beta) to a small group of high-value users.
