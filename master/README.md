# FightIQ Master Pipeline

This folder contains the consolidated, reproducible pipeline for the FightIQ prediction system (v2 Optimized).
It includes all scripts necessary to verify the model's performance, ROI, and robustness.

## Documentation
- **`SCIENTIFIC_PAPER.md`**: A comprehensive report detailing the methodology, experiments, results, and learnings of the project.

## Contents

### Core Components
- **`data/training_data.csv`**: The dataset used for training and verification.
- **`features.json`**: The list of 298 safe, historical features (no leakage).
- **`params.json`**: The optimized hyperparameters found via Optuna (XGBoost + Siamese).
- **`models.py`**: Shared model definitions (Siamese Network architecture).

### Scripts
- **`verify_monte_carlo.py`**: **(Recommended)** Runs 10 random simulations to verify the "Golden Strategy" (Kelly 1/8) on 2025 data.
- **`verify_2024.py`**: Verifies performance on the 2024 "Golden Year" dataset.
- **`verify_roi_2025.py`**: Detailed ROI analysis for 2025 (Flat, Value, Kelly).
- **`verify.py`**: Standard 5-Fold Cross-Validation script (~74.35% accuracy).
- **`train.py`**: Script to train the final production model and save artifacts.

## How to Run

### 1. Verify the "Golden Strategy" (Monte Carlo)
Run this to see the robust performance of the Kelly (1/8) strategy on 2025 data.
```bash
python verify_monte_carlo.py
```
*Expected Result:* Mean ROI ~16%, Mean Bankroll Growth ~6x.

### 2. Verify 2024 Performance
Run this to see the model's performance in 2024.
```bash
python verify_2024.py
```
*Expected Result:* Flat ROI ~23%, Kelly (1/8) Growth ~346x.

### 3. Standard Cross-Validation
Verify the general accuracy of the model.
```bash
python verify.py
```
*Expected Result:* Mean Accuracy ~70.74%.

### 4. Train Final Model
Train the model on the full dataset and generate production artifacts in `models/`.
```bash
python train.py
```

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, torch, joblib

## Key Results
- **2024 ROI (Flat):** 23.47%
- **2025 ROI (Kelly 1/8):** 16.31% (Mean of 10 runs)
- **Optimal Strategy:** Kelly Criterion (1/8 Stake)
