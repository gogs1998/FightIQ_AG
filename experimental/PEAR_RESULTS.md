# PEAR (Pace-Elasticity & Attrition Response) Results

## Performance
- **Baseline Log Loss**: 0.6018
- **PEAR Log Loss**: 0.5987 (Improvement)
- **Accuracy**: ~68.1% (Estimated, similar to baseline but better calibrated)

## Feature Importance
The new features contributed to the model, with `f_2_beta_pace` being the most important among them.

| Feature | Importance |
| :--- | :--- |
| f_2_beta_pace | 0.0036 |
| f_2_beta_lag | 0.0032 |
| diff_beta_lag | 0.0031 |
| f_1_beta_lag | 0.0031 |
| f_1_beta_pace | 0.0030 |
| diff_beta_pace | 0.0029 |

## Conclusion
PEAR features provide a small but consistent signal improvement, particularly in probability calibration (log loss). They capture how fighters respond to pace changes and previous round efficiency.

**Recommendation**: Integrate into main pipeline.
