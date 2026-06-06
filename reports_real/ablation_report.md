# Ablation Report

All rows are prototype-hedger variants evaluated on the held-out test set (lower CVaR is better).

| ablation | cvar_95 | cvar_99 | mean_pnl |
| --- | --- | --- | --- |
| K=4 | 25.26 | 40.04 | 2.596 |
| K=8 | 3.978 | 7.659 | 11.01 |
| K=16 | -0.4348 | 5.206 | 15.21 |
| K=32 | 39 | 41.34 | 23.28 |
| features=greeks_only | 4.453 | 6.52 | 1.371 |
| features=surface_only | 55.39 | 65.85 | 22.49 |

- **K sweep** shows sensitivity to the number of prototypes (interpretability vs capacity).
- **features=greeks_only** removes the volatility surface (scalar Greeks only); **features=surface_only** removes the book Greeks. The gap to the full model quantifies the value of surface-regime information.
