"""Panel-level Greeks and put-call-parity diagnostics for a cleaned option panel.

Thin wrapper over the vectorised Black-Scholes engine in
:mod:`ivsh.pricing.black_scholes`, applied row-wise to a quote panel (one row per
option per day). Used in Phase 4 to attach / validate Greeks alongside any that
the data vendor already supplies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivsh.data.clean import time_to_maturity_years
from ivsh.pricing.black_scholes import bs_greeks

_GREEKS = ("delta", "gamma", "vega", "theta", "vanna", "volga")


def normalize_option_type(s: pd.Series) -> pd.Series:
    """Map vendor flags (C/P, call/put, c/p) to 'call'/'put'."""
    m = {"c": "call", "call": "call", "p": "put", "put": "put"}
    out = s.astype(str).str.strip().str.lower().map(m)
    if out.isna().any():
        bad = sorted(set(s[out.isna()].astype(str)))
        raise ValueError(f"unrecognised option_type values: {bad}")
    return out


def panel_greeks(df: pd.DataFrame, rate: float = 0.0, div: float = 0.0) -> pd.DataFrame:
    """Return ``df`` with delta/gamma/vega/theta/vanna/volga columns added.

    Requires ``spot, strike, iv, option_type`` and a maturity column
    (``ttm_years`` | ``ttm_days`` | ``expiry``).
    """
    df = df.copy()
    ttm = time_to_maturity_years(df)
    opt = normalize_option_type(df["option_type"]).to_numpy()
    spot = df["spot"].to_numpy(dtype=float)
    strike = df["strike"].to_numpy(dtype=float)
    iv = df["iv"].to_numpy(dtype=float)

    for g in _GREEKS:
        df[g] = np.nan
    for kind in ("call", "put"):
        m = opt == kind
        if not m.any():
            continue
        g = bs_greeks(spot[m], strike[m], ttm[m], iv[m], rate, div, kind)
        for name in _GREEKS:
            df.loc[m, name] = g[name]
    return df


def put_call_parity_residual(df: pd.DataFrame, rate: float = 0.0, div: float = 0.0) -> pd.DataFrame:
    """Per (date, strike, expiry) put-call parity residual using mid prices.

    Returns a frame with the residual ``C - P - (S e^{-q tau} - K e^{-r tau})``;
    values far from zero flag mispriced or misaligned quotes.
    """
    need = {"date", "strike", "spot", "option_type", "mid"}
    if not need <= set(df.columns):
        raise ValueError(f"put_call_parity_residual needs columns {sorted(need)}")
    d = df.copy()
    d["ttm_years"] = time_to_maturity_years(d)
    d["_ot"] = normalize_option_type(d["option_type"])
    keys = ["date", "strike", "ttm_years", "spot"]
    calls = d[d["_ot"] == "call"].set_index(keys)["mid"]
    puts = d[d["_ot"] == "put"].set_index(keys)["mid"]
    joined = pd.concat([calls.rename("call"), puts.rename("put")], axis=1).dropna().reset_index()
    tau = joined["ttm_years"].to_numpy(dtype=float)
    fwd_disc = joined["spot"].to_numpy(dtype=float) * np.exp(-div * tau)
    strike_disc = joined["strike"].to_numpy(dtype=float) * np.exp(-rate * tau)
    joined["parity_residual"] = joined["call"] - joined["put"] - (fwd_disc - strike_disc)
    return joined
