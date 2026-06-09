# Data provenance & layout

The real-data study uses **end-of-day option chains** for **SPY** and **QQQ**.
These raw files are large (~3.8 GB total) and are **not committed to git** — they are
downloaded once and regenerated/cleaned by the pipeline. This keeps the repository
lean and reproducible (history was cleaned of the raw blobs; only code, configs,
results, figures, and the paper are tracked).

## Source

- **Vendor:** OptionsDX — <https://www.optionsdx.com/>
  Free historical end-of-day option-chain data (SPY, QQQ, and other US underlyings).
- **Coverage used:** SPY 2010–2023, QQQ 2012–2023 (monthly files).
- **Underlying / rates:** daily adjusted close for the ETF leg; a tenor-matched
  risk-free proxy and dividend/forward estimate (see `configs/`).

## Expected layout

Download the monthly files and place them as follows (the loader accepts either the
raw `.txt` exports or the compressed `.7z` monthly archives):

```text
data/
  raw/
    spy/ spy_eod_YYYYMM.txt        # or spy_eod_YYYY-*.7z monthly archives
    qqq/ qqq_eod_YYYYMM.txt        # or qqq_eod_YYYY-*.7z monthly archives
  interim/                         # cleaned, not-yet-model-ready panels (generated)
  processed/                       # model-ready tensors + backtest datasets (generated)
```

`data/` is git-ignored. Only `data/raw/` needs to be supplied; `interim/` and
`processed/` are produced by the build step.

## Build

```bash
python scripts/build_dataset.py --config configs/data.yaml   # raw -> interim -> processed
# then: make reproduce   (or: python scripts/run_real_data.py)
```

The cleaning funnel (crossed/stale/expired filters, IV bounds, moneyness band,
OTM-only, per-day SVI-denoised surface fit, static no-arbitrage audit) is applied in
the build step and reported to `reports/`. Every dataset version is reproducible
from the configs above.
