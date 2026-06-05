# Interpretable Volatility-Surface Hedger

Prototype-based hedging research stack for option books whose state is an implied-volatility surface rather than spot alone.

## Objective

Build a hedger that maps option-surface regimes to auditable hedge actions. The core idea is to encode each market state as a moneyness-tenor surface tensor, compare it with learned or curated prototypes, and blend prototype hedge actions using similarity weights. This is inspired by ProtoHedge's interpretable policy head, adapted to volatility-surface risk.

## Initial scope

- Surface dataset contract for option chains, implied vols, Greeks, underlying returns, and hedge costs.
- Feature builders for moneyness-tenor surface tensors and regime descriptors.
- Prototype policy interface with similarity-weighted hedge actions.
- Backtest metrics focused on hedge P&L, CVaR, expected shortfall, turnover, and action stability.
- Smoke-testable synthetic workflow until production data is supplied.

## Repo layout

```text
src/ivsh/
  data/          dataset contracts and synthetic surface generator
  features/      surface tensor and regime features
  models/        prototype policy
  backtest/      simple hedging simulator
  evaluation/    risk and stability metrics
configs/         experiment definitions
docs/            research notes and data checklist
scripts/         runnable entry points
tests/           smoke and unit tests
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_synthetic_demo.py
```

## Next decisions

- Underlying universe: SPX/SPY, rates options, single-name options, or futures options.
- Hedge instruments: underlying only, delta-vega basket, listed options, or futures.
- Rebalance cadence and cost model.
- Whether prototypes are learned end-to-end, clustered offline, or seeded from named market regimes.

