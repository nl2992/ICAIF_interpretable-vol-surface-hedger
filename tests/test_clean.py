from __future__ import annotations

import numpy as np
import pandas as pd

from ivsh.data.clean import add_quote_features, clean_option_panel, write_clean_panel
from ivsh.features.greeks import (
    normalize_option_type,
    panel_greeks,
    put_call_parity_residual,
)
from ivsh.pricing.black_scholes import bs_price


def _raw_panel():
    return pd.DataFrame(
        {
            "date": [0, 0, 0, 0, 0, 0],
            "spot": [100.0] * 6,
            "strike": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "ttm_years": [0.25, 0.25, 0.25, -0.1, 0.25, 0.25],
            "option_type": ["C", "C", "C", "C", "C", "C"],
            "bid": [2.0, -1.0, 5.0, 2.0, 2.0, 0.05],
            "ask": [2.2, 1.0, 4.0, 2.2, 2.2, 5.0],  # row2 crossed, row5 huge rel spread
            "iv": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "volume": [10, 10, 10, 10, 0, 10],
            "open_interest": [50, 50, 50, 50, 0, 50],
        }
    )


def test_clean_filters_and_summary():
    clean, summary = clean_option_panel(_raw_panel(), max_rel_spread=0.5, drop_stale=True)
    # removed: bid<0 (row1), crossed (row2), expired (row3), stale zero vol+oi (row4),
    # wide rel spread (row5). Only row0 survives.
    assert len(clean) == 1
    cols = set(summary.table.columns)
    assert {"filter", "removed", "remaining", "pct_removed"} <= cols
    assert summary.table["removed"].sum() == 5


def test_add_quote_features():
    df = add_quote_features(_raw_panel().iloc[[0]].copy(), rate=0.02, div=0.01)
    assert "mid" in df and "ttm_years" in df and "forward" in df and "log_moneyness" in df
    # ATM strike == spot, with carry the forward exceeds spot so log-moneyness < 0
    assert df["log_moneyness"].iloc[0] < 0


def test_panel_greeks_match_engine():
    df = pd.DataFrame(
        {
            "spot": [100.0, 100.0],
            "strike": [100.0, 110.0],
            "ttm_years": [0.5, 0.5],
            "iv": [0.2, 0.25],
            "option_type": ["call", "put"],
        }
    )
    out = panel_greeks(df, rate=0.01, div=0.0)
    assert 0 < out["delta"].iloc[0] < 1  # call delta
    assert -1 < out["delta"].iloc[1] < 0  # put delta
    assert (out["vega"] > 0).all()


def test_normalize_option_type_rejects_garbage():
    assert list(normalize_option_type(pd.Series(["C", "put", "c"]))) == ["call", "put", "call"]
    try:
        normalize_option_type(pd.Series(["X"]))
        assert False, "should have raised"
    except ValueError:
        pass


def test_put_call_parity_residual_near_zero_for_fair_quotes():
    s, tau, r, q = 100.0, 0.5, 0.01, 0.0
    strikes = [90.0, 100.0, 110.0]
    rows = []
    for k in strikes:
        for ot in ("call", "put"):
            price = float(bs_price(s, k, tau, 0.2, r, q, ot))
            rows.append({"date": 0, "spot": s, "strike": k, "ttm_years": tau, "option_type": ot, "mid": price})
    res = put_call_parity_residual(pd.DataFrame(rows), rate=r, div=q)
    assert np.allclose(res["parity_residual"].to_numpy(), 0.0, atol=1e-6)


def test_write_clean_panel(tmp_path):
    clean, _ = clean_option_panel(_raw_panel())
    p = write_clean_panel(clean, str(tmp_path / "interim" / "clean.parquet"))
    assert pd.read_parquet(p).shape[0] == len(clean)
