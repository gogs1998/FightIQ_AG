# Master Props (Prop Hunter)

This directory contains the logic for the **Master Props** model, designed to predict exact fight outcomes (Winner + Method + Round) and identify high-value betting opportunities.

## Key Findings (2024-2025 Backtest)
- **Trifecta Accuracy:** **27.24%** (Winner + Method + Round ALL Correct)
- **ROI:** **48.25%** (Simulated flat betting on all Trifecta predictions)
- **Total Profit:** **$44,100** (from $100 bets over 914 fights)

## Scripts

### 1. `generate_full_predictions.py`
**Purpose:** Generates a comprehensive log of predictions for *every* fight in the 2024-2025 period, regardless of betting thresholds.
**Output:** 
- `full_predictions_2024_2025.csv`: Detailed log of all 914 fights.
- `perfect_predictions_2024_2025.csv`: Subset of 249 fights where the model was perfectly correct.
**Usage:**
```bash
python generate_full_predictions.py
```
*Note: This script trains fresh models on the fly (2010-2023 data) to ensure perfect feature compatibility.*

### 2. `generate_props_log.py`
**Purpose:** Simulates specific betting strategies (e.g., "Bet KO if prob > 50%") and calculates ROI for individual prop types.
**Output:** `props_backtest_2024_2025.csv`
**Usage:**
```bash
python generate_props_log.py
```

## Models
- **Winner:** XGBoost (Boosted parameters)
- **Finish:** XGBoost (Finish vs Decision)
- **Method:** XGBoost (KO vs Submission)
- **Round:** XGBoost (Multi-class 1-5)

## Data
Requires `../master_3/data/training_data_enhanced.csv`.
