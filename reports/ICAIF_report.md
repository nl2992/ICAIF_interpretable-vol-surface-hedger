# Interpretable Volatility-Surface Hedging with Prototype Policies

*Working report / paper draft. All numbers, figures and tables are produced by
the pipeline in this repository (`reports/` = synthetic study, `reports_real/` =
real SPY study) and are reproducible via `make run` and `scripts/run_real_data.py`.*

---

## Abstract

We study whether an **interpretable, prototype-based hedger keyed on the
volatility-surface regime** can reduce tail hedging losses relative to
Black–Scholes delta and delta–vega hedging while remaining competitive with a
black-box deep hedger. Each market state is encoded from the implied-volatility
surface (level, skew, curvature, term slope) together with book and
hedge-instrument Greeks, compared against a small set of learned **prototypes**
(k-means medoids in standardised feature space), and the hedge action is a
similarity-weighted blend of bounded per-prototype actions — so every trade is
traceable to named market regimes. Policies are trained under a cost-adjusted
**CVaR** objective (Rockafellar–Uryasev) with analytic gradients. On a controlled
regime-switching synthetic market the prototype hedger reduces CVaR₉₅ by **53%**
vs delta and **36%** vs delta–vega and edges the black box, with significance
under paired bootstrap and Wilcoxon tests. On **14 years of real SPY options
(2010–2023, 3.5M cleaned quotes)**, run as bounded residuals on the delta–vega
hedge, it beats delta–vega on **every** tail metric (−16% CVaR₉₅, **−61%
max-drawdown**) and on cost-adjusted utility at comparable mean and turnover,
while the flexible black box overfits in-sample drift and **blows up
out-of-sample**. An ablation removing the CVaR term inflates CVaR₉₅ from ~2.4 to
**87.6**, quantifying the value of the risk objective.

---

## 1. Introduction

Standard option hedging keys off spot and scalar Greeks. Yet the *shape* of the
implied-volatility surface — its skew, curvature and term structure, and how they
shift across regimes — carries economically important tail-risk information that
delta/delta–vega hedges ignore. We ask:

> Can an interpretable prototype-based volatility-surface hedger reduce tail hedge
> losses versus delta / delta–vega hedging while staying competitive with a
> black-box deep hedging policy?

Contributions: (i) a cost-aware hedging environment with an analytic P&L gradient;
(ii) an interpretable **prototype action head** over surface regimes; (iii) a
rigorous comparison against deterministic and black-box hedgers under a CVaR
objective on both a controlled synthetic market and 14 years of real SPY options;
(iv) an **anchored residual** formulation that makes learned hedging robust on
real, non-martingale data; (v) a full interpretability + ablation suite.

## 2. Method

![Architecture](figures/architecture.png)

**State.** Per rebalance day we build standardised features: the four surface
factors (level, skew, curvature, term slope), short/long ATM vol and term slope,
realised vol, recent return, one-day level change, and the liability/hedge-option
Greeks (delta, gamma, vega), moneyness and time to maturity. The standardiser is
**fit on training data only** (no leakage).

**Prototype policy.** Prototypes `p_k` are k-means medoids in the standardised
space (fixed before action training). For state `z`, similarity weights are
`w_k = softmax(-‖z − p_k‖² / T)`; the hedge is `a(z) = Σ_k w_k a_k`, where each
`a_k` is a `tanh`-bounded action (underlying shares, hedge-option units). The
top weights, prototype actions and final action are exposed for every decision.

**Anchoring (real data).** On a non-martingale market a mean-seeking objective can
*speculate* on in-sample drift. We therefore run the learned policies as **bounded
residuals on the delta–vega hedge**, `holdings = delta-vega(bank) + a(z)`, so they
remain genuine hedges and can only make modest regime-conditional corrections.

**Objective & training.** We maximise the cost-adjusted CVaR utility
`E[PnL] − λ·CVaR_α(loss)` in the smooth Rockafellar–Uryasev form (auxiliary `η`
optimised jointly), L2-regularised, via L-BFGS-B with **analytic gradients**
(backprop through the policy and the vectorised episode P&L; unit-tested against
finite differences). Validation CVaR drives early stopping.

**Black-box baseline.** A one-hidden-layer MLP with the same inputs, action space,
cost model and objective — the "competitive black box" the prototype is measured
against.

## 3. Data

**Synthetic.** A regime-switching stochastic-vol + jump market emits a parametric
IV surface; it is zero-carry and jump-compensated (a martingale), so a policy can
only improve the objective by genuinely hedging, never by harvesting drift.
Trained Monte-Carlo over many paths; tested on disjoint held-out paths.

**Real.** OptionsDX SPY end-of-day chains, 2010–2023 (`docs/wrds_data_request.md`
documents the original OptionMetrics plan; `docs/data_sources.md` the public
alternatives used here). Cleaning keeps a clean OTM smile (crossed/stale/expired
filters, IV bounds, moneyness band, OTM-only); see `tables/cleaning_funnel.csv`.
The parametric surface is fit per day (SVI-denoised per maturity slice). A static
no-arbitrage audit (monotonicity / butterfly / calendar) is reported in
`arbitrage_audit.md`. **3.5M cleaned OTM quotes across 3,499 trading days.**

## 4. Results

### 4.1 Synthetic market (held-out paths)

![CVaR comparison](figures/cvar_comparison.png)
![P&L distribution](figures/pnl_distributions.png)

The prototype hedger attains the lowest CVaR₉₅/₉₉, worst loss and turnover and the
highest utility (see `tables/model_comparison.csv`): CVaR₉₅ **1.30** vs delta 2.79,
delta–vega 2.02, black box 1.73. Paired bootstrap + Wilcoxon
(`tables/significance.csv`) confirm the prototype's tail is significantly smaller
than delta, delta–vega and the black box (CIs exclude 0, p≈0). Tail loss with
bootstrap CIs is shown in `figures/cvar_ci.png`.

### 4.2 Real SPY options 2010–2023 (chronological, test ≈2020–2023)

The held-out window spans COVID-2020 and the 2022 bear market.

| policy | mean | CVaR₉₅ | CVaR₉₉ | max-DD | turnover | utility |
|---|---|---|---|---|---|---|
| delta | 1.16 | 4.71 | 6.60 | 85.4 | 1,100 | −3.55 |
| delta-vega | 0.95 | 2.84 | 4.75 | 45.6 | 879 | −1.90 |
| black-box MLP | 3.65 | 30.23 | 37.86 | 604.8 | 5,054 | −26.58 |
| **prototype (ours)** | 0.86 | **2.38** | **4.40** | **17.7** | 928 | **−1.52** |

![Cumulative P&L](../reports_real/figures/cumulative_pnl.png)

The interpretable prototype beats delta–vega on every tail metric and on utility
at comparable mean/turnover, while the black box overfits the in-sample drift and
blows up out-of-sample. The cumulative-P&L curve shows the black box's
path-dependent collapse vs the prototype's smooth, low-drawdown profile.

## 5. Ablations

`tables/ablation_metrics.csv` / `ablation_report.md`:

- **K (prototypes):** sweet spot at K=8; too few underfits, too many overfits.
- **Feature set:** full vs greeks-only vs surface-only quantifies surface value.
- **Objective = mean-only (no CVaR):** CVaR₉₅ explodes to **42.4 (synthetic) /
  87.6 (real)** vs ~1.3 / ~2.4 with CVaR — the risk objective is essential.
- **No transaction costs:** isolates the cost drag.
- **Head (prototype vs black box)** and **regime slicing** appear in the main
  comparison and `tail_by_regime.png`.

## 6. Interpretability

![Prototype surfaces](figures/prototype_surfaces.png)
![Prototype actions](figures/prototype_actions.png)
![Activation timeline](../reports_real/figures/activation_timeline.png)

Each prototype reconstructs to a concrete IV surface and a readable hedge action,
and on real data maps to historical regimes (`tables/prototype_catalogue.csv`) —
e.g. a high-level, steep-skew prototype that activates predominantly around the
**February-2018 "volmageddon"**. The activation timeline shows which regime drives
the hedge through the test period; `example_trade.png` audits a single stressed
episode end to end (spot, holdings, prototype weights, cumulative P&L).

## 7. Limitations

- Real-data results are noisier and regime-dependent; over the long horizon the
  optimal anchored residual is conservative (it stays near delta–vega with a small
  systematic vega trim), so the edge is concentrated in the tail/drawdown rather
  than dramatic regime-switching trades. The cleanest comparative finding is
  **black-box overfitting vs prototype robustness**.
- The underlying hedge leg uses SPY (tradable); an SPX study would need a futures
  proxy for the delta leg.
- Costs are proportional half-spread; no market impact / borrow modelling.

## 8. Conclusion

A small, auditable set of volatility-surface prototypes delivers hedging that is
**competitive-to-better than delta–vega on tail risk and cost-adjusted utility**,
clearly more robust out-of-sample than a black-box deep hedger, and **fully
interpretable** — every action traces to named market regimes. The CVaR objective
and (on real data) residual anchoring are the two ingredients that make this work.

## Reproducibility

`experiment_id / dataset_version / model_version / seed / split_id` are recorded in
each `manifest.json`. Synthetic: `make run`. Real: extract OptionsDX archives and
`python scripts/run_real_data.py --data "data/raw/spy/spy_eod_20*.txt" --reports-dir reports_real`.
45 unit tests cover pricing, env P&L identities + analytic gradient, no-lookahead
splits, metrics, SVI, the OptionsDX adapter and the end-to-end pipeline.
