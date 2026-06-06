# WRDS / OptionMetrics Data Request

Exactly what to pull to drive the hedging pipeline on real data. Source:
**WRDS → OptionMetrics → IvyDB US** (library `optionm` on the WRDS cloud). Access
either via the WRDS web query UI or the `wrds` Python package.

## 0. Identifiers (pull this first)

Find the `secid` for the underlyings from the security-name table:

```sql
select secid, ticker, issuer, index_flag
from optionm.secnmd
where ticker in ('SPX', 'SPY');
```

Expected (verify, do not hard-code blindly): **SPX `secid = 108105`**,
**SPY `secid = 109820`**. SPX = European, cash/AM-settled index options
(authoritative). SPY = American ETF options (fully tradable underlying).

## 1. Tables to download

| WRDS table | what it gives | key columns | maps to loader field |
|---|---|---|---|
| `optionm.opprcdYYYY` (Option Prices, per year) | daily option quotes | `secid, date, exdate, cp_flag, strike_price, best_bid, best_offer, volume, open_interest, impl_volatility, delta, gamma, vega, theta, optionid, am_settlement` | `date, expiry(exdate), option_type(cp_flag), strike(strike_price/1000), bid, ask, iv, volume, open_interest` |
| `optionm.secprdYYYY` (Security Prices) | underlying daily level | `secid, date, close, return, cfadj, volume` | `spot (close)` |
| `optionm.zerocd` (Zero-Coupon Yield Curve) | risk-free curve OM used for IV | `date, days, rate` | `rate` (interp to each ttm) |
| `optionm.fwdprdYYYY` (Forward Prices) | implied forward per expiry | `secid, date, expiration, forwardprice, amsettlement` | forward (or derive `div`) |
| `optionm.idxdvd` (Index Dividend Yield) | index dividend yield | `secid, date, expiration, rate` | `div` (for SPX) |
| `optionm.vsurfdYYYY` (Volatility Surface) | OM's smoothed IV on a (delta, maturity) grid | `secid, date, days, delta, cp_flag, impl_volatility, dispersion` | direct surface (fast track) |

Notes:
- **`strike_price` is in dollars × 1000** — divide by 1000.
- `cp_flag` is `C`/`P` → map to `call`/`put`.
- Pull OM's `impl_volatility`/`delta`/`vega` too, to validate our Black-Scholes
  recompute (Phase 4 cross-check).
- To reproduce OM's IV exactly, use **their** `zerocd` rate and `fwdprd` forward.

## 2. Scope to request

- **Universe:** SPX (primary). Add SPY for a fully-tradable robustness check.
- **Date range:** **2007-01-01 → latest** (covers GFC 2008, Aug-2015, Feb-2018
  "volmageddon", COVID 2020, 2022 bear — the regimes the prototype story needs).
  OptionMetrics US starts 1996-01-04 if you want the full history.
- **Expiries:** standard monthly (3rd-Friday, AM-settled SPX) for a clean term
  structure; add weeklies (`SPXW`) only later.
- **Moneyness window:** keep strikes within ~0.5–1.5 × spot to bound file size
  and drop illiquid deep wings.
- **Filter at query time:** `best_bid > 0` and the `secid` to cut volume hard
  (raw SPX option files are ~hundreds of MB/year unfiltered).

## 3. Example pulls (`wrds` Python)

```python
import wrds
db = wrds.Connection(wrds_username="YOUR_ID")

opt = db.raw_sql("""
  select date, exdate, cp_flag, strike_price/1000.0 as strike,
         best_bid, best_offer, volume, open_interest,
         impl_volatility, delta, vega
  from optionm.opprcd2020
  where secid = 108105 and best_bid > 0
""")
und  = db.raw_sql("select date, close as spot from optionm.secprd2020 where secid = 108105")
zero = db.raw_sql("select date, days, rate from optionm.zerocd "
                  "where date between '2020-01-01' and '2020-12-31'")
fwd  = db.raw_sql("select date, expiration, forwardprice from optionm.fwdprd2020 where secid = 108105")
```

Loop the year suffix (`opprcd2007 … opprcd2024`) and concatenate. Save raw to
`data/raw/` as Parquet, one file per table (e.g. `opprcd_spx.parquet`,
`secprd_spx.parquet`, `zerocd.parquet`, `fwdprd_spx.parquet`, `idxdvd_spx.parquet`).

## 4. How it flows into this repo

```
data/raw/*.parquet
  → src/ivsh/data/clean_quotes  (crossed/stale/spread filters, mid, ttm)   [Phase 3]
  → join spot (secprd) + rate (zerocd) + forward/div (fwdprd/idxdvd)
  → ivsh.data.loaders.market_from_option_panel  (per-day surface-factor fit)
  → build_episode_bank  →  ivsh.pipeline.evaluate_and_report
```

`market_from_option_panel` needs at minimum: `date, spot, strike`, an implied vol
(`iv`, or `mid` + `option_type`), and time to maturity (`ttm_years` | `ttm_days`
| `expiry`). `bid`/`ask` power the cleaning filters and the realistic
transaction-cost (half-spread) model.

## 5. Two tracks

- **Track A — fast (recommended first):** use the OptionMetrics **Volatility
  Surface** (`vsurfd`). It is already cleaned and smoothed on a (delta, maturity)
  grid; convert delta→strike (via the supplied delta + BS) and fit our four
  surface factors directly. Gets a real-data result quickly with minimal
  cleaning risk.
- **Track B — rigorous (paper-grade):** raw `opprcd` → `clean_quotes` → per-slice
  **SVI** fit → store `data/processed/surface_tensor.zarr` → project to the
  4-factor surface / feed the SVI surface into `MarketPath`. Full control,
  matches Phase 5.

## 6. Hedge-instrument caveat (decide explicitly)

The SPX **index is not directly tradable**, so the "underlying" hedge leg must be
a proxy — **ES futures** (CME, not in OptionMetrics) or **SPY**. Cleanest
self-consistent tradable set: **SPY options + SPY underlying**. Most authoritative
surface: **SPX options** with the underlying hedge done via ES/SPY (note the
basis). Recommendation: build on SPX for the surface study, hedge the delta leg
with ES front-month (pull ES separately from CME/Datastream) or accept the index
as a frictionless underlying and disclose it.

## 7. Companion data outside OptionMetrics (only if hedging with futures)

- **ES futures** daily settle + bid/ask (CME via Datastream/Refinitiv or
  IB) — for the tradable delta leg if not using SPY.
- Everything else (rates, dividends, forwards, IV) comes from OptionMetrics.
