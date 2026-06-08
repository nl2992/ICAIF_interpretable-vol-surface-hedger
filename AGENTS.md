# AGENTS.md — Interpretable Volatility-Surface Hedger

## Project identity

Repository: `interpretable-vol-surface-hedger`

This project builds an interpretable derivatives hedging system that uses volatility-surface regimes, not only spot prices or scalar Greeks, to make hedging decisions. The core modelling idea is to combine a volatility-surface encoder with a prototype-based action head inspired by ProtoHedge: each hedge action should be traceable to similar historical or learned market regimes.

This file is written for coding/research agents working in this repo. Treat it as the project constitution. Follow it unless a later human instruction explicitly overrides it.

---

## High-level objective

The project seeks to answer:

> Can an interpretable, volatility-surface-aware hedger reduce tail hedge losses versus standard delta or delta-vega hedging, while remaining competitive with a black-box deep hedging policy?

The intended contribution is not merely “ML beats delta.” The contribution should be:

1. A hedger that consumes a structured volatility-surface state.
2. A prototype-based decision layer that makes hedge decisions auditable.
3. A cost-aware hedging simulator with realistic transaction costs and position limits.
4. A rigorous comparison against deterministic hedging baselines and a black-box learned hedger.
5. Evidence that surface-regime information improves tail-risk hedging or interpretability.

The objective is allowed to evolve. If the prototype model does not outperform strong baselines, the project should pivot toward one of these still-valid contributions:

- a robust evaluation framework for surface-aware hedging;
- a surface-regime diagnostic tool for explaining when delta-vega hedging fails;
- a hybrid system where prototypes explain a black-box hedger rather than fully replacing it;
- an arbitrage-aware volatility-surface dataset and benchmark for hedging research.

Do not force a positive result. Document failures clearly.

---

## Research hypothesis

Primary hypothesis:

> Volatility-surface regimes contain economically useful information for hedging tail risk, and a prototype-based action head can preserve much of the performance of a black-box hedger while making hedge actions more interpretable.

Secondary hypotheses:

- Surface features such as skew, curvature, term-structure slope, and front-end vol shocks improve hedge performance relative to scalar spot/Greek states.
- Prototype activation patterns correspond to recognisable regimes, such as front-end vol shock, left-tail skew steepening, parallel vol lift, or calendar inversion.
- A CVaR-oriented objective should reduce downside hedge P&L tails more effectively than a mean-squared-error or variance-only objective.

---

## Non-negotiable principles

1. No look-ahead leakage.
2. All train/validation/test splits must be chronological.
3. Transaction costs must be included in all serious comparisons.
4. A model is not considered better unless it improves net, cost-adjusted, out-of-sample performance.
5. A model is not considered interpretable unless individual actions can be traced to prototypes.
6. Every reported result must include baseline comparisons.
7. Every major result must include robustness or ablation checks.
8. Do not report raw P&L alone; always include tail-risk metrics.
9. Preserve failed experiments. They are useful evidence.
10. Prefer simple baselines before complex models.

---

## First viable product

The first viable project should be deliberately narrow:

- Universe: one liquid index options surface, preferably SPX/SPY.
- Frequency: daily end-of-day.
- Liability: short one representative option, or a small fixed option book.
- Hedge instruments: underlying plus one near-dated ATM option and one medium-tenor ATM option.
- Objective: reduce tail hedge loss and improve cost-adjusted CVaR versus delta/delta-vega hedging.
- Model: surface encoder plus prototype action head.

Do not start with a large multi-underlying options universe. That will hide data and modelling problems.

---

## Data requirements

Required raw data:

- Option chains:
  - timestamp or date;
  - underlying;
  - option type;
  - strike;
  - expiry;
  - bid;
  - ask;
  - mid if available;
  - implied volatility if available;
  - volume;
  - open interest.
- Underlying prices:
  - spot or futures price;
  - returns;
  - realised volatility windows.
- Rates and carry:
  - risk-free curve or tenor-matched proxy;
  - dividend yield or forward/carry estimate.
- Transaction-cost assumptions:
  - underlying bid-ask spread;
  - option bid-ask spread;
  - slippage model;
  - proportional trading cost.
- Hedge instrument mapping:
  - which instruments are tradable on each date;
  - roll rules;
  - liquidity filters.

Optional but useful:

- VIX or volatility-index proxy.
- Macro/rates features.
- Market regime labels.
- Event flags for major stress periods.

---

## Data processing standards

The data pipeline should create three layers:

```text
data/raw/        immutable vendor/API/raw files
data/interim/    cleaned but not model-ready panels
data/processed/  model-ready tensors and backtest datasets
```

Quote cleaning must:

- remove crossed quotes;
- remove non-positive asks;
- remove negative bids;
- remove stale quotes where timestamps allow;
- remove contracts with extreme relative spreads;
- align option quotes to underlying prices;
- compute time to maturity;
- compute forward moneyness or log-moneyness;
- compute mid prices;
- compute or validate implied volatilities;
- compute Greeks consistently.

Surface construction must:

- map each day to a fixed tenor-by-moneyness or tenor-by-delta grid;
- include a missingness mask;
- save raw interpolated surfaces;
- save fitted/smoothed surfaces;
- save surface fit diagnostics;
- flag arbitrage violations.

Every dataset version must be reproducible from configs.

---

## Feature design

Minimum state features:

- volatility surface tensor;
- ATM implied vol by tenor;
- skew by tenor;
- smile curvature by tenor;
- term-structure slope;
- one-day and five-day surface changes;
- liability Greeks;
- hedge instrument Greeks;
- current hedge position;
- underlying returns;
- realised volatility;
- transaction-cost/liquidity proxies;
- time to maturity.

Recommended latent features:

- surface PCA factors;
- surface autoencoder representation;
- regime cluster ID;
- prototype similarity vector;
- recent prototype activation history.

Do not use future realised returns or future volatility in features.

---

## Hedging environment

The environment must simulate sequential hedging:

1. Observe state at rebalance time.
2. Model proposes hedge adjustment.
3. Apply action bounds and position limits.
4. Execute hedge trades with transaction costs.
5. Revalue liability and hedge book at next step.
6. Record pathwise P&L, trades, costs, and exposures.
7. Compute terminal or rolling utility.

The environment must support:

- no-trade baseline;
- delta hedge;
- delta-vega hedge;
- black-box learned hedge;
- prototype learned hedge.

Add tests for:

- zero-cost identities;
- no-trade behaviour;
- transaction-cost calculation;
- settlement logic;
- no look-ahead leakage.

---

## Model architecture

Preferred architecture:

```text
surface/context state
        ↓
surface encoder or temporal MAFN-style encoder
        ↓
latent state z_t
        ↓
prototype similarity layer
        ↓
similarity-weighted bounded hedge action
        ↓
hedging environment
        ↓
cost-adjusted CVaR objective
```

Prototype layer requirements:

- prototypes must be fixed or separately constructed before action training;
- prototypes should be actual medoids or interpretable representative states where possible;
- distances should be computed in standardised latent space;
- output actions must be bounded;
- every inference should expose top prototypes and weights.

Black-box baseline requirements:

- same action space;
- same state information where possible;
- same cost model;
- same train/validation/test split.

---

## Training objective

Default objective:

```text
maximise cost-adjusted utility
with focus on CVaR / expected shortfall of hedge P&L
minus turnover penalty
minus position-limit penalty
```

Report the exact objective used. If CVaR optimisation is unstable, test:

- entropic utility;
- mean-CVaR objective;
- mean-variance objective;
- rolling expected shortfall proxy.

Do not silently change the objective without recording it in the experiment config.

---

## Evaluation standards

Required baselines:

- unhedged;
- Black-Scholes delta hedge;
- delta-vega hedge;
- static regime hedge if easy;
- black-box deep hedger;
- prototype surface hedger.

Required metrics:

- mean P&L;
- median P&L;
- P&L standard deviation;
- 1% CVaR;
- 5% CVaR;
- expected shortfall;
- max drawdown;
- turnover;
- transaction costs;
- cost-adjusted utility;
- utility gap versus black-box;
- tail-loss reduction versus delta/delta-vega;
- prototype activation entropy;
- prototype stability.

Required plots:

- P&L distribution by model;
- tail-loss comparison;
- cumulative hedge P&L;
- turnover over time;
- prototype activation over time;
- prototype surface heatmaps;
- action decomposition for selected trades;
- regime-sliced results.

---

## Robustness and ablation

Run at least these ablations:

- no surface tensor, scalar Greeks only;
- no prototype head, black-box head only;
- no transaction costs;
- no CVaR objective;
- different number of prototypes;
- different hedge instrument menus;
- different rebalance frequencies;
- different market regimes;
- different random seeds.

If a result only holds under one fragile setting, report it as fragile.

---

## Expected contribution if successful

A successful project should contribute:

- a cleaned volatility-surface hedging dataset;
- a cost-aware hedging environment;
- an interpretable prototype-based hedging architecture;
- evidence that surface regimes improve tail hedging or interpretability;
- a reproducible comparison against delta, delta-vega, and black-box learned hedges;
- an audit trail for hedge actions.

---

## Acceptable pivots

If the initial result fails, pivot in the following order:

1. Keep the surface dataset and compare standard hedging policies by regime.
2. Use prototypes as an explanation layer for a black-box hedger.
3. Reduce the hedge action space to underlying-only and prove interpretability first.
4. Replace real options data with simulated surfaces to debug the learning setup.
5. Convert the project into a volatility-surface world-model or surface-regime benchmark.

A negative result is acceptable if it is clean, reproducible, and explains why the prototype method underperformed.

---

## Implementation priorities

Work in this order:

1. Data cleaning.
2. Surface construction.
3. Baseline hedging environment.
4. Deterministic baselines.
5. Black-box baseline.
6. Prototype model.
7. Evaluation and ablations.
8. Interpretability report.
9. Paper/demo write-up.

Do not train the prototype model before the deterministic baselines are validated.

---

## Required commands

The repo should eventually support:

```bash
python scripts/build_dataset.py --config configs/data.yaml
python scripts/train.py --config configs/training.yaml
python scripts/evaluate.py --config configs/backtest.yaml
python scripts/make_report.py --experiment_id <ID>
```

---

## Coding standards

- Prefer clear, typed Python.
- Keep configs outside code.
- Use deterministic seeds.
- Write unit tests for financial accounting logic.
- Avoid notebook-only workflows.
- Keep raw data immutable.
- Save all experiment outputs with config snapshots.
- Log every model comparison in a structured table.
- Use explicit date handling.
- Never assume calendar alignment without checking.

---

## Definition of done

The project is done when:

- cleaned surface data can be rebuilt from raw data;
- baseline hedges run end-to-end;
- prototype hedger trains and evaluates;
- black-box comparison exists;
- transaction costs are included;
- CVaR and expected shortfall are reported;
- prototype decisions are auditable;
- robustness and ablation reports exist;
- final report can be generated from scripts.
