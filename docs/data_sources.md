# Option-Data Sources (WRDS alternatives)

WRDS/OptionMetrics is the gold standard but is currently unavailable. The loader
(`ivsh.data.loaders.market_from_option_panel`) is **source-agnostic** — it only
needs `date, spot, strike`, an implied vol (`iv`, or `mid` + `option_type`), and a
maturity (`ttm_years` | `ttm_days` | `expiry`). Any of the sources below maps to
that contract.

## Recommendation while WRDS is down

**Pivot the working universe to SPY and pull free EOD chains from OptionsDX.**
SPY is directly tradable (cleanest hedge-instrument story), and OptionsDX gives
full historical EOD chains with bid/ask, IV and Greeks for free. Keep SPX as the
aspirational target for when WRDS returns. (Scope note added in
`reports/project_scope.md`.)

## Comparison

| Source | Coverage | History | Cost | Format | Notes |
|---|---|---|---|---|---|
| **OptionsDX** (optionsdx.com) | SPX, SPY, NDX, VIX, equities | EOD, ~2010s→ (more with free account) | **Free** | wide CSV (call+put per strike row) incl. `*_IV`, `*_DELTA`, bid/ask | **Best free option.** Reshape wide→long. |
| **Dolthub** `post-no-preference/options` | US equities/ETFs incl. SPY | daily EOD, multi-year | **Free** | SQL/Dolt; export CSV | Good programmatic free option; no SPX index. |
| **Alpha Vantage** `HISTORICAL_OPTIONS` | US listed (SPY yes; SPX index no) | daily, full chain w/ IV+Greeks | Free key (25 req/day) / premium | JSON/CSV per (symbol, date) | Easy API; free tier slow for bulk → premium for years. |
| **Cboe DataShop** | SPX/SPXW/VIX authoritative | full | Paid | CSV | Authoritative for SPX if budget allows. |
| **ORATS** | US options, smoothed surfaces | full | Trial / academic / paid | API/CSV | Has clean surface product (like vsurfd). |
| **Polygon.io** | US options quotes/aggs | full | Paid tier | API | Good if you already pay. |
| **Databento** | US options | full | Paid (credits) | API | High quality, metered. |
| **yfinance** (Yahoo) | SPY/equity chains | **snapshot only (no history)** | Free | API | Use to *start collecting* daily snapshots going forward. |
| **historicaloptiondata.com** | SPX/SPY/equities | full | Cheap one-time | CSV | Budget bulk history. |
| **IBKR** historical | listed options | account-limited | Account | API | If you trade with IBKR. |

## OptionsDX → loader mapping (wide → long)

OptionsDX rows carry both legs per strike. Typical columns:
`QUOTE_DATE, UNDERLYING_LAST, EXPIRE_DATE, DTE, STRIKE, C_BID, C_ASK, C_IV,
C_DELTA, C_VOLUME, P_BID, P_ASK, P_IV, P_DELTA, P_VOLUME, ...`

Reshape to the loader's long panel:

| loader column | OptionsDX source |
|---|---|
| `date` | `QUOTE_DATE` |
| `spot` | `UNDERLYING_LAST` |
| `strike` | `STRIKE` |
| `expiry` / `ttm_days` | `EXPIRE_DATE` / `DTE` |
| `option_type` | `call` rows from `C_*`, `put` rows from `P_*` |
| `bid` / `ask` | `C_BID`/`C_ASK` (calls), `P_BID`/`P_ASK` (puts) |
| `iv` | `C_IV` (calls), `P_IV` (puts) |
| `volume` | `C_VOLUME` / `P_VOLUME` |

Then: `clean_option_panel → add_quote_features → market_from_option_panel(...,
surface_method="svi") → build_episode_bank → pipeline`.

## Alpha Vantage quick example

```python
import requests, pandas as pd
url = "https://www.alphavantage.co/query"
params = {"function": "HISTORICAL_OPTIONS", "symbol": "SPY",
          "date": "2023-03-15", "apikey": "YOUR_KEY", "datatype": "csv"}
df = pd.read_csv(io.StringIO(requests.get(url, params=params).text))
# df has: contractID, symbol, expiration, strike, type, bid, ask,
#         implied_volatility, delta, gamma, ... -> rename to the loader schema.
```

> A thin `optionsdx_to_panel` / `alphavantage_to_panel` adapter can be added to
> `ivsh.data.loaders` once a source is chosen and a sample file is in hand — the
> column names above are stable enough to wire directly.
