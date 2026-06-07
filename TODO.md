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
- ✅ **Quote cleaning (Phase 3).** `ivsh.data.clean.clean_option_panel`:
  crossed/negative/zero, missing/expired, non-positive IV, abs/rel spread,
  liquidity and stale filters with a per-rule removal summary; mid / ttm /
  forward / log-moneyness features; parquet output to `data/interim/`.
- ✅ **Greeks & parity (Phase 4).** `ivsh.features.greeks`: panel-level
  delta..volga over the BS engine, plus put-call-parity residual diagnostics.
- ✅ **Surface construction (Phase 5).** `ivsh.features.svi`: per-slice SVI
  calibration (`fit_svi_slice`, `fit_svi_day`) and quote denoising
  (`smooth_panel_svi`); `ivsh.data.build_surface`: fixed-grid IV tensor +
  quality metrics (RMSE / max residual) + npz/zarr save. Loader exposes
  `surface_method="svi"`.
- ✅ **Real-data end-to-end (Phase 6 / Step 6).** OptionsDX adapter
  (`optionsdx_to_panel`, `load_optionsdx`), real-data filters (OTM-only, IV
  bounds, moneyness band), `build_data_from_panel` (chronological split),
  `scripts/run_real_data.py`. Ran on SPY 2018–2020 → `reports_real/`.
- ✅ **Anchored residual hedging.** Learned policies as bounded residuals on the
  delta-vega hedge (`TrainConfig.anchor`) so they remain genuine hedges on
  non-martingale real markets.
- ✅ **Date-annotated prototypes (Step 8 / Phase 13).** `prototype_date_annotations`
  maps each prototype to the historical dates/regime it activates on (e.g.
  P6 ≈ Feb-2018 volmageddon).
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
- ✅ **Ablations (Phase 14).** Prototype count K sweep; surface-only vs
  greeks-only vs full feature sets; **no-CVaR (mean-only) objective**; **no
  transaction costs**; plus black-box-vs-prototype (head) and regime slicing in
  the main comparison.
- ✅ **Reproducibility.** `experiment_id / dataset_version / model_version / seed /
  split_id` recorded in `reports/manifest.json`; chronological + held-out-path
  splits; seeded throughout.
- ✅ **Run commands.** One-shot `scripts/run_experiment.py` and the staged
  `build_dataset → train → evaluate → make_report` flow.

## Robustness & generalization (v2 — added 2026-06-07)

The three gaps the v1 paper flagged in its Limitations are now closed for real
(SPY 2010–2023 + QQQ 2012–2023, extracted from the OptionsDX `.7z` archives via
`scripts/extract_data.py`; all artefacts in `reports_real/`):

- ✅ **Second universe (QQQ) + cross-market significance.**
  `scripts/run_real_analysis.py --universe spy=… --universe qqq=…` scores all
  methods per universe and combines per-universe paired-bootstrap results with
  Stouffer's method (`ivsh.evaluation.stats.stouffer_combine`).
- ✅ **Grid search → winning, robust config (v3, 2026-06-07).** The naive anchored
  residual lost to delta–vega on QQQ (combined p≈1e-4 *worse*). Diagnosed
  (`scripts/diagnose_failure.py`: residual adds tail risk in 100% of QQQ stress
  episodes; val 2018–19 calmer than 2020+ test) and fixed via a pre-registered grid
  (`scripts/grid_search.py`, 61 configs × 2 universes, 7 hypotheses, **validation-only
  selection**). Winner = **tail-weighted objective** (cvar_weight=3, α=0.975),
  confirmed once on test (`scripts/confirm_winner.py`): **ties-or-beats delta–vega on
  BOTH SPY (2.34±0.10) and QQQ (5.62±0.63)**; combined p flips 1e-4 worse → **0.079
  favorable**; dominates PPO/SAC by 1–2 orders. Cached banks: `scripts/cache_banks.py`.
  GPU: torch cu124 installed (`RLConfig.device`), CUDA available.
- ✅ **Seminal hero visuals** (`scripts/make_hero_figures.py`): 3D regime-vocabulary
  surfaces, state regime map, robustness landscape, hedge anatomy — integrated into
  `paper/main.tex` (compiles, 7 pp, 0 undefined refs).
- ✅ **Strong deep-RL comparators (PPO + SAC, stable-baselines3 + PyTorch).**
  `src/ivsh/models/deep_rl.py` (`HedgingGymEnv` + `train_sb3`/`evaluate_sb3`; env
  return == `episode_pnl`, unit-tested). *Finding:* PPO/SAC overfit catastrophically
  on this non-martingale data (CVaR₉₅ 35–90); the prototype dominates both by orders
  of magnitude in **both** universes (combined p≈0). They are also the most
  seed-unstable (PPO 52.8±14.6 vs prototype 2.36±0.11).
- ✅ **Walk-forward COVID-2020 fix — volatility-scaled residual cap.**
  `realized_vol_scale` in `ivsh.training.train`, threaded through
  `fit_policy`/`run_policy`. *Finding:* repairs ~70% of the SPY-2020 blow-up
  (59.98 → 17.96) and helps crisis folds broadly (QQQ-2022 10.3 → 3.9), net-beneficial
  across folds, but does **not** reach delta–vega — a mitigation, not a cure.

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
