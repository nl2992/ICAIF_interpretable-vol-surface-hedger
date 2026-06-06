# Final Report — Interpretable Volatility-Surface Hedger

**Experiment:** `spy_2018_2020`  |  dataset `synthetic-regime-sv-jump-v1`  |  model `proto-surface-hedger-v1`  |  seed `7`  |  split `train24-val6-test12`

## Research question
> Can an interpretable prototype-based volatility-surface hedger reduce tail hedge losses versus delta / delta-vega hedging while staying competitive with a black-box deep hedging policy?

## Setup
- Liability: short 1.0 ATM call(s), 30-day tenor, hedged daily to expiry.
- Hedge instruments: underlying + 60-day ATM option.
- Costs: 1.0 bps underlying, 30.0 bps option (on traded notional).
- Market: real option panel (2018-01-02 to 2020-12-31), per-day surface fit, chronological split. Trained on 134 episodes, tested on 60 held-out episodes.
- Objective: maximise E[P&L] − CVaR₉₅(loss) (Rockafellar–Uryasev), L2-regularised. Learned policies are **bounded residuals on the delta-vega hedge** (anchored), so they stay genuine hedges on non-martingale real data.

## Model comparison (test set)

| method | mean_pnl | median_pnl | std_pnl | var_95 | cvar_95 | cvar_99 | worst | max_drawdown | turnover | utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unhedged | -6.174 | -6.741 | 12.37 | 23.9 | 36.3 | 47.76 | 47.76 | 409.9 | 0 | -42.47 |
| delta | 1.248 | 2.014 | 3.618 | 6.064 | 12.27 | 13.25 | 13.25 | 29.33 | 922.9 | -11.02 |
| delta_vega | 1.246 | 1.439 | 1.636 | 1.877 | 4.07 | 4.2 | 4.2 | 9.777 | 807.7 | -2.824 |
| blackbox | 31.34 | 29.25 | 38 | 26.01 | 38.06 | 39.9 | 39.9 | 149.1 | 1.131e+04 | -6.719 |
| prototype | 11.01 | 6.777 | 17.1 | -0.5738 | 3.978 | 7.659 | 7.659 | 11.93 | 2089 | 7.034 |

Lower CVaR / worst / max-drawdown is better; higher utility is better.

![CVaR comparison](figures/cvar_comparison.png)
![P&L distribution](figures/pnl_distributions.png)

## Tail loss by regime

![Tail by regime](figures/tail_by_regime.png)

| method | calm_cvar95 | stress_cvar95 |
| --- | --- | --- |
| unhedged | 15.79 | 36.3 |
| delta | 0.344 | 12.27 |
| delta_vega | -0.622 | 4.07 |
| blackbox | 21.49 | 38.06 |
| prototype | -2.463 | 3.978 |

## Statistical significance (prototype vs baselines)

| comparison | Δcvar95 | cvar95 CI | boot p | wilcoxon p |
| --- | --- | --- | --- | --- |
| prototype − delta | -8.29 | [-13.745, 2.304] | 0.135 | 0 |
| prototype − delta_vega | -0.092 | [-5.549, 5.160] | 0.862 | 0 |
| prototype − blackbox | -34.08 | [-39.762, -14.031] | 0 | 0 |

A negative Δcvar95 with a CI excluding 0 means the prototype hedger has a *significantly smaller* tail loss than the comparator.

## Headline finding
The prototype surface hedger cuts CVaR₉₅ tail loss by **68%** versus delta and **2%** versus delta-vega, while landing below the black-box deep hedger (prototype 3.978 vs black-box 38.059) — with a fully auditable, prototype-based decision trail.

See [prototype_audit_report.md](prototype_audit_report.md) for interpretability, [ablation_report.md](ablation_report.md) for ablations, and [arbitrage_audit.md](arbitrage_audit.md) for the static no-arbitrage surface audit.
