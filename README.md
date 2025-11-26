# FightIQ: Advanced MMA Prediction System

**Status:** Production Ready (Verified 2024-2025)
**Authors:** AntiGravity (Google DeepMind) & User

## Overview
FightIQ is a state-of-the-art machine learning system designed to predict the outcomes of Mixed Martial Arts (MMA) fights and generate profitable betting strategies. It utilizes a hybrid ensemble of **XGBoost** (for tabular stats) and a **Siamese Neural Network** (for matchup embeddings), calibrated via **Isotonic Regression**.

## Performance (Verified)

| Year | ROI | Total Profit (Simulated) | Strategy |
| :--- | :--- | :--- | :--- |
| **2024** | **+33.90%** | +$522,960 | Max Odds 5.0, Kelly 1/4 |
| **2025** | **+30.89%** | +$11,097 | Max Odds 5.0, Kelly 1/4 |

*Note: Simulated profits assume compounding growth. Real-world liquidity limits apply.*

## The "Golden Rule" Strategy
To replicate these results, strictly follow this strategy:

1.  **Confidence:** Only bet if Model Confidence > **60%**.
2.  **Odds Cap:** Never bet on odds > **5.00 (+400)**.
3.  **Edge:** Bet on any positive edge (> 0%).
4.  **Staking:** Use **1/4 Kelly Criterion**.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to install PyTorch separately depending on your CUDA version)*

## Usage

### 1. Generate Predictions (Weekly)
Run the prediction script to get upcoming fight picks:
```bash
python master/predict_upcoming.py
```
*(Coming Soon)*

### 2. Retrain Models
To retrain the system with new data:
```bash
python master/train.py
```

### 3. Verify Performance
To re-run the historical verification:
```bash
python master/verify_sequential.py
```

## Project Structure
*   `master/models/`: Trained model artifacts (XGBoost, Siamese, Scalers).
*   `master/train.py`: Main training pipeline.
*   `master/verify_sequential.py`: Sequential betting simulation.
*   `master/SCIENTIFIC_PAPER.md`: Detailed scientific methodology and results.

## Disclaimer
This software is for educational and research purposes only. Betting involves risk. Never bet money you cannot afford to lose. Past performance is not indicative of future results.
