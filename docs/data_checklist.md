# Data Checklist

Minimum viable dataset:

- Timestamped option chains with bid, ask, mid, implied volatility, strike, expiry, and option type.
- Underlying price and realised return over the hedge interval.
- Greeks from the same model and timestamp, or enough inputs to recompute them.
- Executable hedge instruments and transaction-cost assumptions.
- Filters for stale quotes, crossed markets, missing expiries, and static-arbitrage violations.

