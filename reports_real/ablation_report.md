# Ablation Report

All rows are prototype-hedger variants evaluated on the held-out test set (lower CVaR is better).

| ablation | cvar_95 | cvar_99 | mean_pnl |
| --- | --- | --- | --- |
| K=4 | 2.561 | 5.064 | 0.952 |
| K=8 | 2.383 | 4.397 | 0.8627 |
| K=16 | 6.545 | 9.095 | 1.918 |
| K=32 | 9.534 | 16.1 | 3.804 |
| features=greeks_only | 2.356 | 4.692 | 0.8768 |
| features=surface_only | 2.369 | 4.239 | 0.9367 |
| objective=mean_only | 87.62 | 110.3 | 10.5 |
| no_transaction_costs | 2.212 | 4.209 | 1.026 |

- **K sweep** shows sensitivity to the number of prototypes (interpretability vs capacity).
- **features=greeks_only** removes the volatility surface (scalar Greeks only); **features=surface_only** removes the book Greeks. The gap to the full model quantifies the value of surface-regime information.
- **objective=mean_only** drops the CVaR term (maximise mean P&L only); the resulting rise in tail loss quantifies what the CVaR objective buys.
- **no_transaction_costs** trains and evaluates with zero costs, isolating the cost drag.

Note: the headline model is the full prototype hedger (`K=` the configured value, all features, CVaR objective, with costs) reported in the final report.
