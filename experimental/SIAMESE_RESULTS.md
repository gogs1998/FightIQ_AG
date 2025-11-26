# Siamese Matchup Net & Ensemble Results

## Performance Summary
| Model | Accuracy | Log Loss | Notes |
| :--- | :--- | :--- | :--- |
| **XGBoost Baseline** | 67.77% | 0.6012 | Solid baseline. |
| **Siamese Net (Top 50 Feats)** | 68.42% | 0.7219 | Good accuracy, poor calibration. |
| **Ensemble (50/50)** | **69.88%** | **0.5900** | **BEST RESULT**. Significant improvement in both metrics. |

## Methodology
1.  **Feature Selection**: Used the **Top 50** most important features from the XGBoost model to select symmetric feature pairs (approx 40-50 pairs). This reduced noise compared to using all 263 pairs.
2.  **Siamese Architecture**: Shared MLP encoder (128 hidden units), symmetric loss.
3.  **Ensemble**: Simple 50/50 weighted average of XGBoost and Siamese probabilities.

## Key Insights
- **Synergy**: The Siamese Net and XGBoost make different types of errors. Blending them corrects the Siamese Net's calibration issues while leveraging its superior ranking/comparative ability.
- **Feature Quality**: Filtering to the top 50 features was crucial. Using all features degraded performance (61% acc).
- **Symmetry**: The Siamese architecture naturally enforces symmetry (`P(A>B) = 1 - P(B>A)`), which helps with generalization.

## Recommendation
**DEPLOY IMMEDIATELY.**
1.  This Ensemble approach yields the best model we have seen so far (nearly 70% accuracy).
2.  **Action**: Integrate this into the production pipeline. We need to save the Siamese model and the feature list, and update the prediction API to load both models and ensemble them.
