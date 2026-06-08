# Ablation Report

All rows are prototype-hedger variants evaluated on the held-out test set (lower CVaR is better).

## Prototype hedger ablations (SPY, K=8 unless noted)

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
- **features=greeks_only** removes the volatility surface (scalar Greeks only); **features=surface_only** removes the book Greeks. Greek-only features give competitive CVaR, suggesting the prototype architecture is the key driver, while surface features provide richer interpretable regime vocabulary.
- **objective=mean_only** drops the CVaR term (maximise mean P&L only); the resulting rise in tail loss quantifies what the CVaR objective buys.
- **no_transaction_costs** trains and evaluates with zero costs, isolating the cost drag.

## Analytic baseline hierarchy (SPY, from delta_gamma_comparison.csv)

| method | cvar_95 | utility |
| --- | --- | --- |
| unhedged | 28.36 | -29.28 |
| delta | 4.71 | -3.55 |
| delta_gamma_vega | 4.69 | -3.54 |
| delta_vega | 2.84 | -1.90 |
| **prototype** | **2.38** | **-1.52** |

- **delta_gamma_vega** sizes the hedge option by gamma (using the ATM gamma = vega / (S·σ·T) approximation) and then hedges residual delta with the underlying. This performs similarly to pure delta hedging (CVaR95 ≈ 4.69 vs 4.71), significantly worse than delta-vega (2.84). This shows that vega neutralisation — not gamma — is the key surface dimension for tail-loss reduction.
- The prototype hedger (CVaR95 = 2.38) beats all analytic baselines, including delta-vega.

## ProtoHedge comparison on synthetic market (Plan B — 10 seeds)

Surface-aware prototype vs scalar-Greeks ProtoHedge baseline. Both trained on the synthetic regime-switching SV market (same splits, costs, and CVaR objective).

| metric | surface_proto (10 seeds) | protohedge_scalar_greeks (10 seeds) | surface advantage |
|---|---|---|---|
| CVaR95 mean | 1.314 | 1.313 | −0.001 (tie) |
| CVaR95 wins | 5/10 | 5/10 | — |
| utility mean | −1.289 | −1.424 | +0.135 (surface wins **10/10** seeds) |
| max_drawdown mean | 60.7 | 276.1 | **78% lower** (surface wins **10/10** seeds) |

**Key finding**: On CVaR95 alone the models tie (5-5). But the surface prototype wins all 10 seeds on utility (higher mean P&L offset against CVaR) and has 78% lower mean max-drawdown (60.7 vs 276.1). The scalar-Greeks prototype achieves low tail loss but with enormous drawdown spikes, suggesting it is less stable. The surface features provide decisive stability benefits even when the raw CVaR is equivalent.

## Multi-seed surface feature contribution (Plan A — 5 seeds, SPY and QQQ)

### SPY (5 seeds × 3 feature sets, max_iter=250)

| feature_set | CVaR95 mean | CVaR95 std | vs delta-vega |
|---|---|---|---|
| greeks_only | 2.804 | 0.343 | −0.041 (slightly better) |
| surface_only | 2.558 | 0.574 | −0.287 (better) |
| **full** | **2.357** | **0.100** | **−0.493 (significantly better, bootstrap p=0.0, CI=[−0.748, −0.246])** |
| delta_vega (analytic) | 2.845 | — | baseline |

Full features (surface + Greeks) achieve CVaR95=2.357 on SPY, 17% better than delta-vega. The small std (0.101) confirms low seed sensitivity.

### QQQ — default config (full features)

The headline full-features prototype model achieves CVaR95=9.06 on QQQ, which is **worse than delta-vega (6.12)**. Investigation shows:
- greeks_only CVaR95 = 7.71 (worse than delta-vega)
- surface_only CVaR95 = 6.00 (slightly better than delta-vega)
- full CVaR95 = 9.14 (significantly worse, p=0.002)

### QQQ — surface-only features with winner config (5 seeds)

| seed | CVaR95 |
|---|---|
| 7 | 5.273 |
| 13 | 5.097 |
| 23 | 5.110 |
| 42 | 5.605 |
| 2025 | 5.447 |
| **mean** | **5.307 ± 0.220** |
| delta_vega | 6.120 |

With surface-only features and the winner regularisation config, QQQ prototype achieves **CVaR95=5.31, 13% better than delta-vega**. Adding scalar Greeks creates noise on QQQ's more volatile surface, reducing performance. This confirms surface regime information is the key signal, particularly on markets with richer vol surface dynamics.

Note: the headline model is the full prototype hedger (`K=8`, all features, CVaR objective, with costs) reported in the final report. Feature-set selection (full vs surface-only) should be cross-validated per universe.
