# Profit-Aware Loss & Staking Results

## Model Metrics
| Model | Accuracy | Log Loss |
| :--- | :--- | :--- |
| Baseline | 0.6874 | 0.6015 |
| Profit-Weighted | 0.6154 | 0.6668 |

## Betting Simulation (Holdout Set)
| Model           | Strategy    |        Bankroll |      ROI |
|:----------------|:------------|----------------:|---------:|
| Baseline        | Flat (5%)   |     8.79511e+06 | 0.38463  |
| Baseline        | Kelly (1/4) |     1.10744e+10 | 0.12325  |
| Profit-Weighted | Flat (5%)   | 10006.4         | 0.117728 |
| Profit-Weighted | Kelly (1/4) |     1.48095e+08 | 0.424687 |

## Interpretation
Best Strategy: **Profit-Weighted + Kelly (1/4)** with ROI **42.47%**.
