# 🗺️ FightIQ V2 Roadmap: The "Quantopian for MMA" Platform
**Vision:** Move beyond a simple "Prediction Service" to a full-stack **Algorithmic Trading Platform for MMA**. We empower users to build, test, and deploy their own betting strategies using our leak-free data and infrastructure.

---

## 🏗️ Phase 0: The Foundation (Data Ingestion)
**Goal:** Automate the data pipeline so the platform runs 24/7 without manual intervention.
*   **Automated Scrapers:** Daily jobs to fetch new fight announcements and results.
*   **Odds Stream:** Real-time connection to Odds API for live line movement.
*   **Data Warehouse:** Moving from CSVs to a proper SQL Database (PostgreSQL) for multi-user access.

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

---

## 🧪 Phase 2: The "Strategy Lab" (User-Driven Discovery)
**Goal:** Gamify the research process. Let users feel like Quants discovering the "Holy Grail".

### 2.1 The "Lego Block" System
*   **Concept:** We don't give them the answer. We give them the blocks.
*   **Feature Library:** Users select *which* variables they believe in:
    *   "I think `Reach` matters most." -> Selects `diff_reach`.
    *   "I think `Age` is key." -> Selects `diff_age`.
*   **The Hook:** They build a strategy, backtest it, and see *their* ROI.
*   **Tiered Access:**
    *   **Free:** Basic stats (Wins, Losses).
    *   **Pro:** Advanced stats (Striking Differential, Elo).
    *   **Whale:** "God Mode" features (Boruta Selected, Genetic Optimization).

### 2.2 The Backtest Loop
*   User clicks "Simulate".
*   System runs their custom strategy on 2010-2025 data.
*   **Result:** "Your Strategy ROI: +12%". (Vs FightIQ Benchmark: +30%).
*   **Psychology:** This proves FightIQ's value while keeping them engaged trying to beat us.

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
