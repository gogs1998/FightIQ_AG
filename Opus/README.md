# Gold Model - Production Pipeline

## Verified Results: **72.65% Accuracy** (Leak-Free)

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Model Accuracy** | 72.65% |
| **Favorite Baseline** | 66.67% |
| **Edge Over Baseline** | +6.0% |
| **Test Fights** | 618 (2024-2025) |

### Leakage Tests: ALL PASSED ✅

| Test | Result |
|------|--------|
| Feature Name Scan | ✅ No forbidden terms |
| Correlation Check | ✅ Max 0.362 |
| Random Labels | ✅ 51.0% |
| Random Features | ✅ 51.8% |

### Features (8 Total)

1. `ip` - Implied probability (1/odds)
2. `ipd` - Implied probability difference
3. `wr` - Win rate difference
4. `slpm` - Strikes per minute difference
5. `sdef` - Striking defense difference
6. `tdef` - Takedown defense difference
7. `rwdiff` - Recent wins (last 5) difference
8. `sdiff` - Win streak difference

### Optimal Betting Strategy

**Best ROI: 108.7%**
- Min Edge: 10%
- Min Confidence: 70%
- Underdog Only: Yes
- 32 bets, 81.2% win rate

**Best Balanced: 88.7% ROI**
- Min Edge: 8%
- Min Confidence: 55%
- Underdog Only: Yes
- 79 bets, 73.4% win rate

### Files

```
gold/
├── data/
│   └── UFC_full_data_silver.csv
├── models/
│   └── (trained models)
├── results/
│   ├── betting_log_2024_2025.csv
│   └── strategy_results.csv
├── train.py
├── predict.py
└── README.md
```

### Quick Start

```bash
# Verify model
cd ../platinum
python gemini_style_tests.py

# Train
cd ../gold
python train.py

# Backtest betting
cd ../platinum
python grid_search_roi.py
```
