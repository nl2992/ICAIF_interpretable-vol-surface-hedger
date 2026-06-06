# Project Scope

Phase-1 decisions, now committed (these gate the data request). The v1 results in
`final_report.md` are on a synthetic market that mirrors this scope; real-data
validation follows the same contract.

## Committed decisions

| decision | choice |
|---|---|
| **Universe** | SPX options (primary, European/cash-settled, authoritative). SPY options (American ETF) as a fully-tradable robustness check. |
| **Observation frequency** | Daily end-of-day. |
| **Liability** | Short one ATM option, ~30-day tenor, held to expiry. |
| **Hedge instruments** | Underlying + **two listed ATM options** (a near-dated and a longer-dated), giving control over delta, vega and term/gamma exposure. |
| **Rebalance cadence** | Daily (weekly as a robustness check). |
| **Objective** | Maximise cost-adjusted CVaR utility `E[P&L] − CVaR₉₅(loss)`, preserving interpretability via volatility-surface prototypes. |
| **Prototype learning** | End-to-end (prototypes from clustering, actions + temperature learned under the CVaR objective) — as implemented. |
| **Primary comparison** | Unhedged · delta · delta-vega · black-box deep hedger · prototype surface hedger. |
| **Data source** | WRDS OptionMetrics IvyDB US (see `docs/wrds_data_request.md`). |

## Underlying / tradability note

The SPX index is not directly tradable; the delta-hedge "underlying" leg is a
proxy (ES futures or SPY). For a fully self-consistent tradable set, run the SPY
variant. The SPX variant uses the authoritative surface and discloses the
index/futures basis. This is an explicit modelling choice, not an oversight.

## Evaluation protocol

Train on a Monte-Carlo ensemble (synthetic) or rolling in-sample windows (real);
evaluate on disjoint held-out paths / future windows. Significance via paired
bootstrap CIs and the Wilcoxon signed-rank test on per-episode P&L; metrics
sliced by start regime (calm vs stress). Real-data prototypes are additionally
annotated with the historical dates they activate on (e.g. "Prototype k ≈ the
March-2020 vol regime").
