# FightIQ Master 3: Engineering Specifications

## 1. System Architecture

### 1.1 High-Level Data Flow
```mermaid
graph TD
    A[Raw Data (CSV)] --> B[Feature Engineering (generate_features.py)]
    B --> C[Enhanced Data (CSV)]
    C --> D[Training Pipeline (train.py)]
    D --> E[XGBoost Model]
    D --> F[Siamese Network]
    D --> G[LSTM Sequence Encoder]
    E & F & G --> H[Ensemble Logic]
    H --> I[Predictions (Probability)]
```

### 1.2 Tech Stack
*   **Language**: Python 3.8+
*   **Core Libraries**: `pandas` (Data), `numpy` (Math), `torch` (Deep Learning), `xgboost` (Gradient Boosting).
*   **Optimization**: `boruta` (Feature Selection), `optuna` (Hyperparams).

## 2. Feature Engineering Specifications

### 2.1 PEAR (Pace-Elasticity & Attrition Response)
*   **Source**: `features/pear.py`
*   **Logic**: Regression of `SigStrDiff` vs `OpponentPace` per round.
*   **Output**: `beta_pace` (Slope), `beta_lag` (Intercept).
*   **Purpose**: Measures cardio efficiency and ability to handle pressure.

### 2.2 Chin Health Decay
*   **Source**: `features/chin.py`
*   **Logic**: `ChinScore_t = ChinScore_{t-1} * RecoveryFactor - DamageReceived`.
*   **Params**: `recovery_rate=0.95`, `ko_penalty=2.0`.
*   **Purpose**: Models cumulative neurological damage.

### 2.3 Dynamic Elo
*   **Source**: `features/dynamic_elo.py`
*   **Logic**: Standard Elo with K-factor multipliers for finish type (KO/Sub) and round number.
*   **Purpose**: Opponent-adjusted strength rating.

## 3. Model Architecture

### 3.1 LSTM Sequence Encoder (Phase 6)
*   **Input**: Tensor `(Batch, Seq_Len=5, Input_Dim=11)`.
*   **Features**: `[odds, elo, age, reach, height, win_streak, etc.]`
*   **Layer**: `LSTM(input_size=11, hidden_size=32, num_layers=1)`.
*   **Output**: Last hidden state `h_n` (Size 32).
*   **Integration**: Concatenated with static features in the Siamese Network.

### 3.2 Siamese Matchup Network
*   **Input**: `(FighterA_Stats, FighterB_Stats, SeqA_Embed, SeqB_Embed)`.
*   **Structure**:
    *   `Branch`: `Linear(Input -> 64) -> ReLU -> BatchNorm -> Dropout(0.3)`.
    *   `Merge`: `abs(BranchA - BranchB)`.
    *   `Head`: `Linear(64 -> 1) -> Sigmoid`.
*   **Loss Function**: Symmetric Binary Cross Entropy.
*   **Training**: 50 Random Seeds (Best Seed Selection).

### 3.3 Ensemble
*   **Method**: Weighted Average.
*   **Formula**: `P_final = w * P_xgb + (1-w) * P_siamese`.
*   **Current Weight**: `w_xgb ≈ 0.15`, `w_siamese ≈ 0.85`.

## 4. Validation Protocol

### 4.1 Walk-Forward Validation
*   **Method**: Rolling Window (Train < Year X, Test = Year X).
*   **Years**: 2020, 2021, 2022, 2023, 2024.
*   **Metric**: Accuracy, LogLoss, ROI (Value Sniper).

### 4.2 Leakage Prevention
*   **Feature Scan**: Automated check for terms `winner`, `result`, `outcome`.
*   **Time Travel**: All features use `shift(1)` or pre-fight averages.
*   **Monkey Tests**: Random Labels/Features must yield ~Baseline Accuracy.

## 5. Deployment Requirements
*   **Input**: `upcoming.csv` (Must contain `f_1_name`, `f_2_name`, `event_date`).
*   **Dependencies**: `training_data_enhanced.csv` (For historical context/sequences).
*   **Artifacts**:
    *   `models/xgb_master3.pkl`
    *   `models/finish_master3.pkl`
    *   `models/siamese_master3.pth`
    *   `features_selected.json`
    *   `params.json`
