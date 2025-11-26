# AFM (Adversarial Fragility Margin) Results

## Performance
- **Baseline Log Loss**: 0.6018
- **AFM Log Loss**: 0.6038 (Degradation)
- **Accuracy**: 67.45% (Degradation)

## Feature Importance
The `AFM_score` feature had a low importance of **0.00275**.

| Feature | Importance |
| :--- | :--- |
| AFM_score | 0.00275 |

## Conclusion
AFM failed to improve the model. The hypothesis that "fragility" to small stat changes predicts upsets did not hold up with this simple proxy implementation. It added noise rather than signal.

**Recommendation**: **Discard**. Do not integrate.
