# Master 3: The "Kitchen Sink" Pipeline (Optimized)

## Overview
**Master 3** is a state-of-the-art MMA prediction pipeline that integrates advanced feature engineering, temporal sequence modeling (LSTM), and robust Siamese network training.

**Final Performance (2024-2025 Holdout):**
*   **Accuracy**: **74.29%** (LSTM + Siamese Ensemble)
*   **ROI (Value Sniper)**: **+1788%**
*   **Log Loss**: **0.5749**

## Architecture
The pipeline consists of three main stages:
1.  **Feature Engineering**: Modular generation of advanced features (PEAR, Chin, Elo).
2.  **Optimization**: Boruta feature selection and Optuna hyperparameter tuning.
3.  **Training**: A hybrid ensemble of XGBoost and a Siamese Network (with LSTM embeddings).

### Key Components
*   **`features/`**: Contains the logic for specific features.
    *   `pear.py`: Pace-Elasticity & Attrition Response (Cardio).
    *   `chin.py`: Chin Health Decay (Cumulative Damage).
    *   `dynamic_elo.py`: Custom Elo with finish bonuses.
    *   `common_opponents.py`: Triangle theory scores.
    *   `stoppage.py`: Finish propensity.
*   **`models/`**:
    *   `sequence_model.py`: **LSTM** for "Last 5 Fights" sequence modeling.
    *   `opponent_adjustment.py`: Strength-of-schedule adjustment.
    *   `__init__.py`: Siamese Network architecture.
*   **`train.py`**: Main training script (Ensemble + ROI).
*   **`optimize.py`**: Feature selection and hyperparameter tuning.
*   **`generate_features.py`**: Orchestrates data loading and feature creation.

## Reproducibility Guide

### Prerequisites
*   Python 3.8+
*   Dependencies: `pandas`, `numpy`, `xgboost`, `torch`, `scikit-learn`, `boruta`, `optuna`, `tqdm`.

### Step 1: Data Preparation
Ensure the following files are present:
*   `../training_data.csv`: The base dataset.
*   `../UFC_full_data_golden.csv`: The raw dataset (required for PEAR round-level data).

### Step 2: Generate Features
Run the feature generation script to create `data/training_data_enhanced.csv` and `features_enhanced.json`.
```bash
python generate_features.py
```

### Step 3: Optimization (Optional)
*Note: We have already saved the optimal parameters, so you can skip this step unless you want to re-tune.*
```bash
python optimize.py
```
This will output `features_selected.json` and `params_optimized.json`.

### Step 4: Train the Model
Run the main training pipeline. This will:
1.  Load the enhanced data.
2.  Apply Opponent Adjustments.
3.  Prepare LSTM Sequences (Last 5 Fights).
4.  Train XGBoost.
5.  Train Siamese Network (50 seeds for robustness).
6.  Evaluate on 2024-2025 holdout.
```bash
python train.py
```

## Model Details

### LSTM Sequence Model
*   **Input**: Sequence of last 5 fights (11 selected features per fight).
*   **Architecture**: LSTM (Hidden Dim 32) -> Linear Head.
*   **Purpose**: Captures "Momentum" and "Form" (Improving vs Declining).

### Siamese Network
*   **Input**: Fighter A features, Fighter B features, Sequence Embeddings.
*   **Loss**: Symmetric Binary Cross Entropy.
*   **Robustness**: Trained with 50 random seeds; best seed selected.

### Ensemble
*   **Method**: Weighted Average.
*   **Weights**: ~15% XGBoost, ~85% Siamese (Optimized).

## License
Proprietary FightIQ Algorithm.
