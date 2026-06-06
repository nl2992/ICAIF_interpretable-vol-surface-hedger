# Ablation Report

All rows are prototype-hedger variants evaluated on the held-out test set (lower CVaR is better).

| ablation | cvar_95 | cvar_99 | mean_pnl |
| --- | --- | --- | --- |
| K=4 | 1.383 | 1.746 | -0.1118 |
| K=8 | 1.298 | 1.759 | 0.028 |
| K=16 | 1.643 | 2.526 | 0.1361 |
| K=32 | 1.837 | 2.669 | 0.2407 |
| features=greeks_only | 1.344 | 1.753 | -0.1309 |
| features=surface_only | 1.605 | 1.956 | -0.118 |
| objective=mean_only | 42.36 | 58.78 | 1.147 |
| no_transaction_costs | 1.429 | 1.97 | 0.2007 |

- **K sweep** shows sensitivity to the number of prototypes (interpretability vs capacity).
- **features=greeks_only** removes the volatility surface (scalar Greeks only); **features=surface_only** removes the book Greeks. The gap to the full model quantifies the value of surface-regime information.
- **objective=mean_only** drops the CVaR term (maximise mean P&L only); the resulting rise in tail loss quantifies what the CVaR objective buys.
- **no_transaction_costs** trains and evaluates with zero costs, isolating the cost drag.

Note: the headline model is the full prototype hedger (`K=` the configured value, all features, CVaR objective, with costs) reported in the final report.
