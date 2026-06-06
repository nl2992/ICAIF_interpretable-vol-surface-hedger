# Roadmap & Status — Interpretable Volatility-Surface Hedger

Status legend: ✅ implemented · 🟡 partial · ⬜ future work.

The v1 release is a **self-contained synthetic-market study** that realises the
full scientific core end-to-end (data → train → evaluate → report). Phases that
require proprietary option data are scaffolded but deferred.

## Core (implemented)

- ✅ **Market & data.** Regime-switching stochastic-vol + jump simulator with a
  parametric IV surface, martingale (zero-carry, jump-compensated) so there is no
  drift to speculate on. Monte-Carlo train paths, disjoint held-out test paths.
  (`src/ivsh/data/market.py`)
- ✅ **Pricing & Greeks.** Vectorised Black-Scholes price + delta/gamma/vega/
  theta/vanna/volga, implied-vol solver. (`src/ivsh/pricing/black_scholes.py`)
- ✅ **Hedging environment.** Episode-based, daily rebalancing, transaction costs
  on traded notional, fully vectorised P&L **and analytic P&L gradient** (unit-
  tested vs finite differences). No-trade P&L identity verified.
  (`src/ivsh/envs/hedging_env.py`)
- ✅ **Features.** Surface factors, term/skew/curvature, realised vol, book and
  hedge-instrument Greeks; leak-free standardisation fit on train only.
- ✅ **Baselines.** Unhedged, delta, delta-vega. (`src/ivsh/baselines/`)
- ✅ **Black-box deep hedger.** numpy MLP, same inputs/actions/costs/objective.
  (`src/ivsh/models/blackbox.py`)
- ✅ **Prototype surface hedger.** k-means prototypes + learned bounded action per
  prototype + similarity temperature; interpretable similarity-weighted action.
  (`src/ivsh/models/prototype_policy.py`)
- ✅ **Training.** Cost-adjusted CVaR utility (Rockafellar–Uryasev), analytic
  gradients, L-BFGS-B, validation early stopping. (`src/ivsh/training/`)
- ✅ **Evaluation.** Mean/median/std, VaR, CVaR₉₅/₉₉, worst, max drawdown,
  turnover, utility; **paired bootstrap CIs + Wilcoxon**; regime-sliced metrics.
  (`src/ivsh/evaluation/`)
- ✅ **Interpretability.** Prototype catalogue, reconstructed prototype surfaces,
  latent embedding, activation entropy, **example-trade audit**.
- ✅ **Ablations.** Prototype count K sweep; surface-vs-greeks-only feature sets.
- ✅ **Reproducibility.** `experiment_id / dataset_version / model_version / seed /
  split_id` recorded in `reports/manifest.json`; chronological + held-out-path
  splits; seeded throughout.
- ✅ **Run commands.** One-shot `scripts/run_experiment.py` and the staged
  `build_dataset → train → evaluate → make_report` flow.

## Deferred (future work)

- ✅ Real option-chain ingestion & cleaning loader (`ivsh.data.loaders`): CSV/Parquet
  reader, crossed/stale/expired/wide-spread filters, and a per-day parametric
  surface fit that maps real quotes into the same `MarketPath`/`EpisodeBank`.
- ✅ Static no-arbitrage diagnostics (`ivsh.features.arbitrage`): strike monotonicity,
  butterfly/convexity (implied density >= 0), and calendar (total-variance)
  checks; emitted as `reports/arbitrage_audit.md`.
- ⬜ SVI / arbitrage-free surface fitting on observed quotes.
- 🟡 Additional baselines (delta-gamma-vega with two options, static-regime,
  historical-regression hedge).
- ⬜ Temporal surface encoder over rolling windows; auxiliary surface-reconstruction loss.
- ⬜ Intraday cadence; multi-option liability books.
- ⬜ Further ablations (no-CVaR / mean-variance objective, no-cost, action-menu sweep).

## Acceptance criteria

- ✅ Dataset leak-free (held-out paths + train-only standardisation + purged splits).
- ✅ Surface construction reproducible (seeded parametric model).
- ✅ Baselines run · ✅ black-box runs · ✅ prototype hedger runs.
- ✅ Backtest chronological / held-out · ✅ CVaR metrics · ✅ transaction costs.
- ✅ Prototype audit trail · ✅ final report with comparison, ablation, example trade.
