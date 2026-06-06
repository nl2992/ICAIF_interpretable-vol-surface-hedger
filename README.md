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
pytest                       # 22 tests, ~4s

# One-command experiment (data -> train -> evaluate -> report), ~3 min:
python scripts/run_experiment.py --config configs/experiment.yaml
```

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

Expected panel columns: `date, spot, strike`, an implied vol (`iv`, or `mid` +
`option_type` to imply it), and time to maturity (`ttm_years` | `ttm_days` |
`expiry`). Full contract in [`docs/data_checklist.md`](docs/data_checklist.md).
To run the whole experiment on real data, build train/val/test banks from disjoint
date ranges and pass them to `ivsh.pipeline.evaluate_and_report`.

See [`TODO.md`](TODO.md) for the full research roadmap and the status of each phase.
