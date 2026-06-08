# Four-Universe Grid Notes

The four-universe grid extends the earlier SPY/QQQ search to include the newly
cached SLV and gap-safe SPX banks:

```text
python scripts/grid_search.py --universe spy qqq slv spx --procs 10 --out reports_real/grid_four_universe
```

The run evaluated 61 configurations across 4 universes, for 244 fits. SPX is
built as two contiguous histories, 2010-2016 and 2018-2023, so no episode bridges
the missing 2017 data gap.

## Pre-Registered Raw-Excess Selection

The original grid selection rule minimises mean validation excess CVaR in raw P&L
units. On the four-universe panel this is dominated by SPX's larger index-level
P&L scale. It selected `H3_K12`:

- SPY test excess vs delta-vega: +0.893
- QQQ test excess vs delta-vega: +2.217
- SLV test excess vs delta-vega: +0.011
- SPX test excess vs delta-vega: -6.774

This is not a cross-market win.

## Normalised Selection Diagnostic

Because SPX has a much larger P&L scale, a fair cross-market rule should compare
relative excess:

```text
(prototype_CVaR95 - delta_vega_CVaR95) / abs(delta_vega_CVaR95)
```

The normalised validation-selection table is written to
`grid_four_universe/grid_selection_normalized.csv`. Even under this diagnostic,
the validation-selected config does not tie-or-beat all four markets on test.

## Exploratory Finding

The best diagnostic test config is `H4_rl21000.0`, a strong do-no-harm residual
shrinkage policy. It tie-or-beats delta-vega on all four markets:

- mean relative test excess: -3.3%
- max relative test excess: +0.48%
- tie-or-beat count: 4 / 4

This should be framed as exploratory because it is selected after inspecting
test results. It is nevertheless a useful next hypothesis: conservative residual
shrinkage appears to be the most transferable mechanism across SPY, QQQ, SLV,
and SPX.

## Practical Takeaway

The satisfying result is not "the same prototype configuration wins everywhere."
The cleaner statement is:

1. Surface features add economically meaningful tail reduction on SPX, though
   current bootstrap confidence is wide.
2. Tuned PPO remains catastrophically worse than the prototype and delta-vega on
   SPY.
3. Thin-surface SLV does not support rich prototype regimes; it needs conservative
   shrinkage toward delta-vega.
4. The strongest cross-market candidate is a do-no-harm residual policy, which
   should be pre-registered and tested on future data or a held-out fifth market.
