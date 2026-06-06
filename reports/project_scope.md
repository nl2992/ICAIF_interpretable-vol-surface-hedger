# Project Scope (v1)

**Universe.** Synthetic index-style options on a single underlying (a regime-
switching stochastic-vol + jump market). Designed so the *shape* of the implied
surface — level, skew, curvature, term slope, and their regime — carries
tail-risk information beyond spot.

**Observation frequency.** Daily end-of-day surface snapshots.

**Liability.** Short one ATM option (30-day tenor), held to expiry.

**Hedge instruments.** Underlying + one longer-dated (60-day) ATM option.

**Hedge cadence.** Daily rebalancing to expiry; costs on traded notional
(1 bp underlying, 30 bp option).

**Objective.** Maximise cost-adjusted CVaR utility `E[P&L] − CVaR₉₅(loss)` while
preserving interpretability through volatility-surface prototypes.

**Primary comparison.** Unhedged · delta · delta-vega · black-box deep hedger ·
prototype surface hedger (ours).

**Evaluation protocol.** Train on a Monte-Carlo ensemble of market paths; report
on a disjoint held-out set of paths. Significance via paired bootstrap CIs and
the Wilcoxon signed-rank test on per-episode P&L; metrics also sliced by start
regime (calm vs stress).

**Data note.** No proprietary data is required; the synthetic market is the
experimental testbed. The contract for substituting real option chains is in
`docs/data_checklist.md`.
