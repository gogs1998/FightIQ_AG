# Final Ensemble Results (Hybrid Approach)

## Performance Summary
| Model | Accuracy | Log Loss | Notes |
| :--- | :--- | :--- | :--- |
| **XGBoost (Elo + PEAR)** | 67.53% | 0.5981 | Good calibration, slightly lower accuracy than baseline. |
| **Siamese Net (Original Top 50)** | 69.15% | 0.7425 | Excellent accuracy/ranking, poor calibration. |
| **Ensemble (50/50)** | **69.88%** | **0.5848** | **NEW BEST**. Best Accuracy & Best Log Loss. |

## Methodology
1.  **XGBoost Component**: Trained on Elo features + **PEAR features** (Pace-Elasticity). This model provides excellent probability calibration (Log Loss < 0.60).
2.  **Siamese Component**: Trained on **Original Top 50** features (from baseline XGBoost). This ensures the Siamese Net focuses on the strongest, most robust signals (Elo, Odds, etc.) without being diluted by experimental features that might add noise to the ranking task.
3.  **Ensemble**: Simple 50/50 weighted average.

## Key Insights
- **PEAR Works**: Adding PEAR to the XGBoost component improved the overall Ensemble Log Loss from 0.5900 to 0.5848.
- **Hybrid Feature Selection**: Using different feature sets for different components was key.
    - XGBoost benefits from PEAR (calibration).
    - Siamese benefits from a focused, high-signal feature set (Original Top 50).
- **Synergy**: The ensemble achieves ~70% accuracy with very low log loss, making it highly deployable.

## Recommendation
**DEPLOY THIS HYBRID ENSEMBLE.**
1.  **Train & Save**:
    - `ufc_model_pear.pkl` (XGBoost with PEAR)
    - `siamese_model.pth` (Siamese with Top 50)
    - `features_pear.json` (Feature list for XGBoost)
    - `features_top50.json` (Feature list for Siamese)
2.  **Pipeline**:
    - Calculate PEAR features for new fights.
    - Run both models.
    - Average predictions.
