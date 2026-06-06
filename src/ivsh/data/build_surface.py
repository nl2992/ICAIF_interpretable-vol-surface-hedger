"""Build and persist a fixed-grid implied-volatility tensor — Phase 5.

Samples a :class:`~ivsh.data.market.MarketPath` (synthetic or loaded/SVI-fitted)
onto a fixed (moneyness, tenor) grid, giving a ``[days, tenors, moneyness]``
tensor plus coordinate metadata. Saved as ``.npz`` by default (dependency-free);
if a ``.zarr`` path is requested and ``zarr`` is installed it is used instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ivsh.data.market import TRADING_DAYS, MarketPath


def build_surface_tensor(
    market: MarketPath,
    moneyness: tuple[float, ...] | None = None,
    tenor_days: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, dict]:
    """Return ``(tensor[days, tenors, moneyness], coords)``."""
    cfg = market.config
    moneyness = np.asarray(moneyness or cfg.grid_moneyness, dtype=float)
    tenor_days = np.asarray(tenor_days or cfg.grid_tenor_days, dtype=int)
    n = market.n_days
    tensor = np.empty((n, len(tenor_days), len(moneyness)))
    for d in range(n):
        spot = market.spot[d]
        strikes = moneyness * spot
        for ti, tnr in enumerate(tenor_days):
            tensor[d, ti] = market.iv(d, strikes, d + int(tnr))
    coords = {
        "days": np.arange(n),
        "tenor_days": tenor_days,
        "moneyness": moneyness,
    }
    return tensor, coords


def surface_quality(market: MarketPath, panel) -> dict:
    """RMSE / max residual of the fitted surface vs observed quotes.

    ``panel`` must have columns ``date, spot, strike, iv`` and a maturity column;
    ``date`` is matched to the market's day index by sorted order.
    """
    import pandas as pd

    from ivsh.data.clean import time_to_maturity_years

    df = panel.copy()
    df["_day"] = pd.factorize(df["date"], sort=True)[0]
    ttm = time_to_maturity_years(df)
    day = df["_day"].to_numpy()
    strike = df["strike"].to_numpy(dtype=float)
    obs = df["iv"].to_numpy(dtype=float)
    expiry_day = day + np.round(ttm * TRADING_DAYS).astype(int)
    fit = market.iv(day, strike, expiry_day)
    resid = fit - obs
    return {
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "max_abs_resid": float(np.max(np.abs(resid))),
        "n": int(len(resid)),
    }


def save_surface_tensor(
    tensor: np.ndarray, coords: dict, path: str = "data/processed/surface_tensor.npz"
) -> str:
    """Persist the tensor + coords. Uses zarr for ``.zarr`` paths if available."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".zarr"):
        try:
            import zarr  # type: ignore

            root = zarr.open_group(path, mode="w")
            root["iv"] = tensor
            for k, v in coords.items():
                root[k] = np.asarray(v)
            return path
        except ImportError:
            path = path[:-5] + ".npz"
            p = Path(path)
    np.savez_compressed(p, iv=tensor, **{k: np.asarray(v) for k, v in coords.items()})
    return path
