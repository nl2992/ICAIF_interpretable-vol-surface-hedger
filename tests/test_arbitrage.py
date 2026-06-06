from __future__ import annotations

import numpy as np

from ivsh.data.market import MarketConfig, MarketPath
from ivsh.features.arbitrage import audit_market


def _flat_market(n=5, level=0.2, skew=0.0, curv=0.0, slope=0.0) -> MarketPath:
    cfg = MarketConfig(n_days=n, rate=0.0, div=0.0)
    return MarketPath(
        config=cfg,
        days=np.arange(n),
        spot=np.full(n, 100.0),
        level=np.full(n, level),
        skew=np.full(n, skew),
        curv=np.full(n, curv),
        slope=np.full(n, slope),
        regime=np.zeros(n, dtype=int),
        realized_vol=np.full(n, level),
        log_return=np.zeros(n),
        rate=0.0,
        div=0.0,
    )


def test_flat_surface_is_arbitrage_free():
    rep = audit_market(_flat_market())
    assert rep.summary["butterfly_pct"] == 0.0
    assert rep.summary["calendar_pct"] == 0.0
    assert rep.summary["monotonicity_pct"] == 0.0
    assert rep.summary["days_flagged"] == 0


def test_excess_curvature_flags_butterfly():
    # An extremely convex smile drives the implied density negative at the wings.
    fine = tuple(round(x, 3) for x in np.linspace(0.7, 1.3, 13))
    rep = audit_market(_flat_market(curv=40.0), moneyness=fine)
    assert rep.summary["butterfly_pct"] > 0.0


def test_inverted_term_structure_flags_calendar():
    rep = audit_market(_flat_market(slope=-0.12))  # total variance falls with tenor
    assert rep.summary["calendar_pct"] > 0.0


def test_report_has_per_day_table():
    rep = audit_market(_flat_market())
    assert {"day", "regime", "monotonicity", "butterfly", "calendar"} <= set(rep.per_day.columns)
    assert len(rep.per_day) == 5
