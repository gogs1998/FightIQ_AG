# REST (Ref/Commission Stoppage Tolerance) Results

## Performance
- **Baseline Log Loss**: 0.6018
- **REST Log Loss**: 0.6033 (Degradation)
- **Accuracy**: 67.37% (Degradation)

## Feature Importance
Only the referee multiplier showed any importance. Location-based priors (state, country) had 0.0 importance.

| Feature | Importance |
| :--- | :--- |
| REST_ref_mult | 0.0023 |
| REST_state_mult | 0.0000 |
| REST_country_mult | 0.0000 |

## Conclusion
REST features failed to improve the model. While referee bias showed some signal, it wasn't enough to outweigh the noise or complexity added. Location priors were ineffective.

**Recommendation**: **Discard**. Do not integrate.
