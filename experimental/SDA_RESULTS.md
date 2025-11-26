# SDA (Strategic Diversity & Adaptability) Results

## Performance
- **Baseline Log Loss**: 0.6018
- **SDA Log Loss**: 0.6000 (Improvement)
- **Accuracy**: 67.61% (Drop from ~68.1%)

## Feature Importance
SDA features, particularly position entropy, showed significant importance.

| Feature | Importance |
| :--- | :--- |
| f_2_sda_pos_entropy | 0.0039 |
| diff_sda_target_entropy | 0.0035 |
| f_2_sda_target_entropy | 0.0033 |
| f_1_sda_target_js | 0.0028 |
| f_1_sda_target_entropy | 0.0027 |
| diff_sda_pos_entropy | 0.0027 |

## Conclusion
SDA features improved Log Loss (better probability calibration) but hurt raw Accuracy. This suggests they add valuable signal about uncertainty or match dynamics but might introduce noise for simple winner prediction.

**Recommendation**: 
- Consider including for probability models (betting).
- Further investigation needed to see if they can be combined with PEAR without hurting accuracy.
