# SLV Experiment Notes

SLV was added as a lower-correlation third universe with a visibly thinner option
surface than SPY/QQQ. The data were extracted from local OptionsDX archives only;
no GLD archive was present locally.

## Data Build

- Extracted `data/slv_*.7z` to `data/raw/slv/`.
- Built `artifacts/bank_slv.pkl` with `scripts/cache_banks.py --thin-surface`.
- Cleaner profile: wider moneyness band, looser relative-spread cap, and no
  minimum volume floor, because SLV has fewer usable strikes.
- Result: 450,546 clean quotes, 660 hedging episodes, 2016-2023.

## Confirmed Winner Transfer

The pre-registered SPY/QQQ tail-weighted winner was evaluated on SLV without
retuning. It did not beat delta-vega:

- prototype CVaR95: 0.220 +/- 0.058
- delta-vega CVaR95: 0.155
- paired-bootstrap delta CVaR95: +0.028, p=0.158

This should be reported as a transfer miss / statistical near-tie, not a win.

## SLV-Only Exploratory Grid

`scripts/grid_search.py --universe slv --procs 4 --out reports_real/grid_slv`
found that 19 of 61 configs tie-or-beat delta-vega on SLV test. The strongest
test performer was `H4_rl2100.0`, a do-no-harm residual shrinkage setting:

- test CVaR95: 0.151
- delta-vega CVaR95: 0.155
- test excess: -0.0038

Because this config is identified after inspecting SLV test performance, treat it
as exploratory. A clean next step is to pre-register a cross-universe selection
rule that prefers validation-improving configs with residual shrinkage and low
seed dispersion, then re-confirm on a held-out universe or future-year fold.

## Interpretation

The SLV prototype audit shows prototype collapse: one high-vol/front-end-inverted
prototype owns roughly 76% of test episodes. This is consistent with a thin
surface whose sparse strikes do not support the same regime vocabulary as SPY or
QQQ. For SLV, the useful learned behavior appears to be conservative shrinkage
toward delta-vega rather than richer surface-regime action selection.
