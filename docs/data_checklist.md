# Data Checklist

Minimum viable dataset:

- Timestamped option chains with bid, ask, mid, implied volatility, strike, expiry, and option type.
- Underlying price and realised return over the hedge interval.
- Greeks from the same model and timestamp, or enough inputs to recompute them.
- Executable hedge instruments and transaction-cost assumptions.
- Filters for stale quotes, crossed markets, missing expiries, and static-arbitrage violations.

## Loader contract (`ivsh.data.loaders`)

`market_from_option_panel(df, rate, div)` consumes a **long-form quote panel**
(one row per option per day) and returns a `MarketPath` that drops straight into
`build_episode_bank`. Required and optional columns:

| column | required | notes |
|---|---|---|
| `date` | yes | sortable trading-day key (int index or datetime); defines the day grid |
| `spot` | yes | underlying price on that day |
| `strike` | yes | option strike |
| `iv` | one of | implied volatility; **or** provide `mid` + `option_type` to imply it |
| `ttm_years` / `ttm_days` / `expiry` | one of | time to maturity (years / calendar days / expiry date) |
| `bid`, `ask` | optional | enable `clean_quotes` spread/crossed filters and `mid` |
| `option_type` | optional | `"call"`/`"put"`, needed only when imputing `iv` from `mid` |
| `volume`, `open_interest` | optional | for liquidity filtering |

Pipeline: `load_option_panel` (CSV/Parquet) → `clean_quotes` (crossed / negative /
expired / wide-spread filters, returns a per-rule removal summary) →
`market_from_option_panel` (per-day OLS fit of level / skew / curvature / term
slope) → `build_episode_bank`. The fitted parametric surface is the bridge that
lets real data reuse the identical environment, features and models.

Notes:
- The day grid is taken from distinct `date` values; use a regular business-day
  panel. Time-to-maturity is annualised consistently with the env (`/252`).
- The `regime` label (for evaluation slicing only) is derived causally from the
  trailing-median vol level — it is never used as a model input, so it cannot
  leak future information into training.
- For a full real-data experiment, build disjoint date-range banks for
  train/val/test and call `ivsh.pipeline.evaluate_and_report`.

