"""Static no-arbitrage diagnostics for an implied-volatility surface.

Three classic checks per trading day, evaluated on a (moneyness, tenor) grid of
European call prices implied by the surface:

* **Monotonicity in strike** — call price must be non-increasing in strike
  (slope dC/dK <= 0).
* **Butterfly / convexity** — call price must be convex in strike (the implied
  risk-neutral density is non-negative); the strike-slopes must be
  non-decreasing.
* **Calendar** — total implied variance ``w = iv^2 * tau`` must be
  non-decreasing in tenor at fixed log-moneyness.

These flag (rather than remove) violations, mirroring how a real surface is
audited. The synthetic market is arbitrage-light but can drift into mild
violations in extreme states, so the audit is informative there too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ivsh.data.market import TRADING_DAYS, MarketPath
from ivsh.pricing.black_scholes import bs_price


@dataclass
class ArbitrageReport:
    per_day: pd.DataFrame  # one row per day with violation counts + cell totals
    summary: dict[str, float]

    def to_markdown(self) -> str:  # pragma: no cover - cosmetic
        s = self.summary
        return (
            "# Arbitrage Audit\n\n"
            f"Grid cells audited: {int(s['n_cells'])} "
            f"({int(s['n_days'])} days x {int(s['n_tenors'])} tenors x {int(s['n_moneyness'])} strikes).\n\n"
            f"- Strike-monotonicity violations: **{s['monotonicity_pct']:.3f}%** of slope checks\n"
            f"- Butterfly / convexity violations: **{s['butterfly_pct']:.3f}%** of convexity checks\n"
            f"- Calendar (total-variance) violations: **{s['calendar_pct']:.3f}%** of calendar checks\n\n"
            f"Days with any violation: {int(s['days_flagged'])} / {int(s['n_days'])}.\n"
        )


def audit_market(
    market: MarketPath,
    moneyness: tuple[float, ...] | None = None,
    tenor_days: tuple[int, ...] | None = None,
    tol: float = 1e-6,
) -> ArbitrageReport:
    """Audit a market path's surface for static-arbitrage violations."""
    cfg = market.config
    moneyness = moneyness or cfg.grid_moneyness
    tenor_days = tenor_days or cfg.grid_tenor_days
    m = np.asarray(moneyness, dtype=float)
    tdays = np.asarray(tenor_days, dtype=float)
    rate, div = market.rate, market.div

    rows = []
    mono_v = bfly_v = cal_v = 0
    mono_n = bfly_n = cal_n = 0
    days_flagged = 0

    for d in range(market.n_days):
        spot = market.spot[d]
        strikes = m * spot  # [M]
        ttm = tdays / TRADING_DAYS  # [T]
        # Price call grid C[T, M] and total variance w[T, M].
        C = np.empty((len(tdays), len(m)))
        W = np.empty_like(C)
        for ti, tnr in enumerate(tdays):
            iv = market.iv(d, strikes, d + tnr)
            C[ti] = bs_price(spot, strikes, ttm[ti], iv, rate, div, "call")
            W[ti] = iv**2 * ttm[ti]

        # Monotonicity + butterfly per tenor (across strike).
        dK = np.diff(strikes)  # [M-1]
        slopes = np.diff(C, axis=1) / dK  # [T, M-1]
        d_mono = int((slopes > tol).sum())
        d_bfly = int((np.diff(slopes, axis=1) < -tol).sum())
        mono_v += d_mono
        bfly_v += d_bfly
        mono_n += slopes.size
        bfly_n += max(np.diff(slopes, axis=1).size, 0)

        # Calendar per moneyness (across tenor).
        d_cal = int((np.diff(W, axis=0) < -tol).sum())
        cal_v += d_cal
        cal_n += np.diff(W, axis=0).size

        flagged = d_mono + d_bfly + d_cal
        days_flagged += int(flagged > 0)
        rows.append(
            {
                "day": d,
                "regime": int(market.regime[d]),
                "monotonicity": d_mono,
                "butterfly": d_bfly,
                "calendar": d_cal,
            }
        )

    per_day = pd.DataFrame(rows)
    n_cells = market.n_days * len(tdays) * len(m)
    summary = {
        "n_days": market.n_days,
        "n_tenors": len(tdays),
        "n_moneyness": len(m),
        "n_cells": n_cells,
        "monotonicity_pct": 100.0 * mono_v / max(mono_n, 1),
        "butterfly_pct": 100.0 * bfly_v / max(bfly_n, 1),
        "calendar_pct": 100.0 * cal_v / max(cal_n, 1),
        "days_flagged": days_flagged,
    }
    return ArbitrageReport(per_day=per_day, summary=summary)
