# Interpretable Volatility-Surface Hedger

An interpretable, **prototype-based** option-hedging agent whose state is the
**implied-volatility surface** (its regime), not just spot, delta, or a scalar
Greek. Each market state is encoded as a moneyness–tenor surface, compared with a
small set of learned market *prototypes*, and the hedge action is a
similarity-weighted blend of the prototypes' learned actions — so every trade is
traceable to named volatility regimes.

> **Research question.** Can an interpretable prototype-based volatility-surface
> hedger reduce tail hedge losses versus delta / delta-vega hedging while staying
> competitive with a black-box deep-hedging policy?

**Answer (this repo, synthetic study): yes.** On held-out market paths the
prototype hedger cuts CVaR₉₅ tail loss by ~50% versus delta and ~35% versus
delta-vega, and **beats** the black-box deep hedger on tail risk and turnover —
while remaining fully auditable. All differences are significant under paired
bootstrap and Wilcoxon tests.

| policy | mean P&L | CVaR₉₅ | CVaR₉₉ | worst | turnover | utility |
|---|---|---|---|---|---|---|
| unhedged | −0.40 | 13.21 | 18.19 | 23.49 | 0 | −13.61 |
| delta | −0.15 | 2.79 | 4.22 | 5.12 | 299 | −2.93 |
| delta-vega | −0.22 | 2.02 | 3.35 | 5.10 | 245 | −2.24 |
| black-box MLP | +0.19 | 1.73 | 2.74 | 4.05 | 270 | −1.54 |
| **prototype (ours)** | **+0.03** | **1.30** | **1.76** | **2.24** | **175** | **−1.27** |

*(test set; lower CVaR / worst / turnover is better, higher utility is better. Reproduce with the command below.)*

### Real data — SPY 2010–2023 (3,499 trading days; train ≈2010–18, test ≈2020–23)

On 14 years of real OptionsDX SPY options (3.5M cleaned OTM quotes, surface fit per
day, chronological split, learned policies run as **bounded residuals on the
delta-vega hedge** so they stay genuine hedges on a non-martingale market). The
held-out test window spans COVID-2020 and the 2022 bear market:

| policy | mean P&L | CVaR₉₅ | CVaR₉₉ | max-DD | turnover | utility |
|---|---|---|---|---|---|---|
| delta | 1.16 | 4.71 | 6.60 | 85.4 | 1,100 | −3.55 |
| delta-vega | 0.95 | 2.84 | 4.75 | 45.6 | 879 | −1.90 |
| black-box MLP | 3.65 | 30.23 | 37.86 | 604.8 | 5,054 | −26.58 |
| **prototype (ours)** | 0.86 | **2.38** | **4.40** | **17.7** | 928 | **−1.52** |

What is **statistically significant** (paired bootstrap, `tables/significance.csv`):
the prototype hedger ≫ delta (Δcvar₉₅ −2.3, p≈0) and ≫ the black-box deep hedger
(−27.8, p≈0) — the constrained, auditable policy **generalises out-of-sample while
the flexible black box overfits in-sample drift and blows up** (CVaR₉₅ 30, 5k
turnover). Versus the strongest classical baseline, **delta-vega, the tail
improvement is directional but not significant** (Δcvar₉₅ −0.46, 95% CI
[−0.93, +0.08], bootstrap p=0.086) — on the tail it is a slight-but-not-proven
edge / statistical tie, with better max-drawdown and utility. On this long real
sample the learned residual is small (activation entropy ≈0.11): the prototype
essentially **reproduces delta-vega with a conservative trim** rather than learning
dramatically different regime actions (the rich regime-specific behaviour shows up
on the synthetic market). Ablation: dropping the CVaR term explodes the tail to
CVaR₉₅ **87.6**. Full report set in [`reports_real/`](reports_real/); see
[`reports/ICAIF_report.md`](reports/ICAIF_report.md) for the write-up.

## Method

- **Market.** Regime-switching stochastic-volatility + jump model that emits a
  *parametric* IV surface (level, skew, curvature, term slope) whose shape — not
  just spot — carries tail-risk information. Zero carry, jump-compensated, so the
  market is a martingale: a policy can only improve the objective by genuinely
  hedging, never by harvesting drift. Trained Monte-Carlo over many paths; tested
  on disjoint held-out paths.
- **Liability & hedges.** Short an ATM option, hedged daily to expiry with the
  underlying plus a longer-dated ATM option. Transaction costs charged on traded
  notional (build, rebalance, and terminal liquidation).
- **Objective.** Maximise `E[P&L] − CVaR₉₅(loss)` in the smooth
  Rockafellar–Uryasev form, L2-regularised. Both learnable policies are trained
  with **analytic gradients** (exact backprop through the policy and the P&L) via
  L-BFGS-B, with validation early stopping.
- **Prototype policy.** Prototypes are k-means medoids in standardised feature
  space; the policy learns a bounded hedge action per prototype and a similarity
  temperature. The hedge is the softmax-similarity-weighted average of prototype
  actions — the exact quantity exposed for interpretability.
- **Baselines.** Unhedged, Black–Scholes delta, delta-vega, plus a black-box MLP
  deep hedger sharing the same features, action space, costs and objective.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # numpy / pandas / scipy / matplotlib / pyyaml
pytest                       # 25 tests, ~4s

# One-command experiment (data -> train -> evaluate -> report), ~3 min:
python scripts/run_experiment.py --config configs/experiment.yaml
```

Or use the Makefile: `make install`, `make test`, `make run` (full),
`make run-fast` (no ablations), `make staged` (four-stage flow), or
`make reproduce` (install + test + run from a clean checkout).

The repository already ships the generated study artifacts so results are
viewable without running anything: the markdown reports, figures and CSV tables
under [`reports/`](reports/) and the trained prototype model at
`checkpoints/proto_surface_hedger_best.npz`.

This writes [`reports/final_report.md`](reports/final_report.md) (model
comparison + significance tests), [`reports/prototype_audit_report.md`](reports/prototype_audit_report.md)
(prototype catalogue, surfaces, example-trade audit) and
[`reports/ablation_report.md`](reports/ablation_report.md), plus figures and CSV
tables.

### Staged pipeline (matches the project roadmap)

```bash
python scripts/build_dataset.py --config configs/experiment.yaml   # -> artifacts/dataset.pkl
python scripts/train.py         --config configs/experiment.yaml   # -> artifacts/models.pkl + checkpoint
python scripts/evaluate.py      --config configs/experiment.yaml   # -> reports + tables + figures
python scripts/make_report.py   --experiment_id ivsh_demo          # -> summary of the run
```

## Repo layout

```text
src/ivsh/
  data/        market simulator (regime SV + jumps) and IV surface
  pricing/     vectorised Black-Scholes price & Greeks (delta..volga)
  features/    surface tensor + leak-free standardisation
  envs/        episode-based hedging environment (vectorised P&L + analytic grad)
  models/      prototype hedger, black-box MLP, k-means
  baselines/   unhedged / delta / delta-vega Greek hedges
  training/    CVaR objective + L-BFGS training with analytic gradients
  evaluation/  metrics, paired bootstrap / Wilcoxon, backtest, report & figures
  utils/       chronological splits, feature selection
  pipeline.py  end-to-end orchestration (build_data -> train -> evaluate -> report)
configs/       experiment.yaml (canonical config)
scripts/       run_experiment.py + staged build/train/evaluate/make_report
reports/       generated report, figures, tables
tests/         pricing, env identities, no-lookahead, metrics, pipeline smoke
```

## Notes & scope

- **Data.** The study runs on a self-contained synthetic market (no proprietary
  options data required), the standard setting for deep-hedging methodology
  papers. **Real option chains plug in via `ivsh.data.loaders`** — see below.
- **Dependencies.** Pure numpy/scipy/pandas/matplotlib — no PyTorch — so it runs
  anywhere (including Python 3.14). Gradients are hand-derived and unit-tested
  against finite differences.
- **Reproducibility.** Every run records `experiment_id`, `dataset_version`,
  `model_version`, `seed` and `split_id` in `reports/manifest.json`.

## Using real option data

Real surfaces flow through the **exact same** environment, features, baselines and
models — the bridge is the parametric surface (level / skew / curvature / term
slope), which `ivsh.data.loaders` fits per trading day by least squares to your
cleaned implied-vol quotes:

```python
from ivsh.data.loaders import load_option_panel, clean_quotes, market_from_option_panel
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank

panel = load_option_panel("data/raw/spx_chains.parquet")   # long-form quotes
panel, summary = clean_quotes(panel)                        # crossed/stale/expired filters
market = market_from_option_panel(panel, rate=0.0, div=0.0) # -> MarketPath
bank = build_episode_bank(market, EnvConfig())              # -> drop into the pipeline
```

**OptionsDX SPY (used for the real run above):**

```bash
# raw monthly files extracted under data/raw/spy/
python scripts/run_real_data.py \
    --data "data/raw/spy/spy_eod_2018*.txt" "data/raw/spy/spy_eod_2019*.txt" \
           "data/raw/spy/spy_eod_2020*.txt" \
    --reports-dir reports_real --surface svi
```

`ivsh.data.loaders.optionsdx_to_panel` reshapes the wide call+put chains to the
long panel; the driver cleans to a clean OTM smile, fits the SVI surface, and runs
the anchored study.

Expected panel columns: `date, spot, strike`, an implied vol (`iv`, or `mid` +
`option_type` to imply it), and time to maturity (`ttm_years` | `ttm_days` |
`expiry`). Full contract in [`docs/data_checklist.md`](docs/data_checklist.md).
To run the whole experiment on real data, build train/val/test banks from disjoint
date ranges and pass them to `ivsh.pipeline.evaluate_and_report`.

The committed research scope is in [`reports/project_scope.md`](reports/project_scope.md),
the exact WRDS / OptionMetrics pull list is in
[`docs/wrds_data_request.md`](docs/wrds_data_request.md), and free/cheap
alternatives (OptionsDX, Dolthub, Alpha Vantage, …) with column mappings are in
[`docs/data_sources.md`](docs/data_sources.md). Surfaces can be denoised per
maturity with SVI via `market_from_option_panel(..., surface_method="svi")`.

See [`TODO.md`](TODO.md) for the full research roadmap and the status of each phase.
