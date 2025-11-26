# CST (Counterfactual Style Transport) Results

## Performance
- **Baseline Log Loss**: 0.6018
- **CST Log Loss**: 0.6033 (Degradation)
- **Accuracy**: 67.53% (Degradation)

## Feature Importance
Most transported features had **0.0 importance**. The only features with non-zero importance were the "effective sample size" weights, suggesting the *similarity* of past opponents matters more than the transported stats themselves.

| Feature | Importance |
| :--- | :--- |
| f_1_CST_weight_n_eff | 0.0036 |
| f_2_CST_weight_n_eff | 0.0026 |
| f_1_CST_sig_strikes_landed | 0.0000 |
| f_2_CST_sig_strikes_landed | 0.0000 |

## Conclusion
CST failed to provide predictive value and added significant computational cost. The hypothesis that we can "transport" performance from past opponents to the current one did not hold up with this implementation.

**Recommendation**: **Discard**. Do not integrate into the main pipeline.
