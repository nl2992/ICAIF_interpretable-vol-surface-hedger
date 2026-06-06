"""SVI (Stochastic-Volatility-Inspired) smile calibration — Phase 5.

Per maturity slice we fit Gatheral's *raw* SVI parametrisation of total implied
variance ``w = iv^2 * tau`` as a function of forward log-moneyness ``k``:

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

with ``b >= 0``, ``|rho| < 1``, ``sigma > 0``. This is the industry-standard
stable smile model. We use it to **denoise** raw quotes slice by slice before the
cross-maturity surface-factor fit, which sharpens the surface that feeds the
hedging environment. A butterfly (convexity) sanity check is provided via the
existing arbitrage module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2))

    def iv(self, k: np.ndarray, tau: float) -> np.ndarray:
        w = np.maximum(self.total_variance(k), 1e-10)
        return np.sqrt(w / max(tau, 1e-10))


def _svi_w(k: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, b, rho, m, sigma = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def fit_svi_slice(k: np.ndarray, total_var: np.ndarray, tau: float) -> tuple[SVIParams, float]:
    """Fit one maturity slice; returns (params, RMSE in implied-vol units)."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_var, dtype=float)
    if k.size < 5:
        raise ValueError("need >= 5 points to fit 5 SVI parameters")

    wmax = max(float(w.max()), 1e-6)
    p0 = np.array([max(float(w.min()), 1e-4) * 0.5, 0.1, -0.3, 0.0, 0.1])
    lb = np.array([0.0, 0.0, -0.999, k.min() - 1.0, 1e-4])
    ub = np.array([1.5 * wmax, 5.0, 0.999, k.max() + 1.0, 2.0])
    p0 = np.clip(p0, lb, ub)
    res = least_squares(lambda p: _svi_w(k, p) - w, p0, bounds=(lb, ub), max_nfev=2000)
    params = SVIParams(*res.x)
    fit_iv = np.sqrt(np.maximum(_svi_w(k, res.x), 1e-10) / max(tau, 1e-10))
    obs_iv = np.sqrt(np.maximum(w, 1e-10) / max(tau, 1e-10))
    rmse = float(np.sqrt(np.mean((fit_iv - obs_iv) ** 2)))
    return params, rmse


def fit_svi_day(
    df_day: pd.DataFrame, rate: float = 0.0, div: float = 0.0, min_quotes: int = 5
) -> pd.DataFrame:
    """Fit SVI to every maturity slice of one day; return per-slice diagnostics.

    Expects columns ``spot, strike, iv`` and ``ttm_years``.
    """
    rows = []
    spot = df_day["spot"].to_numpy(dtype=float)
    for tau, g in df_day.groupby("ttm_years"):
        if len(g) < min_quotes:
            continue
        s = float(g["spot"].iloc[0])
        fwd = s * np.exp((rate - div) * float(tau))
        k = np.log(g["strike"].to_numpy(dtype=float) / fwd)
        w = g["iv"].to_numpy(dtype=float) ** 2 * float(tau)
        try:
            params, rmse = fit_svi_slice(k, w, float(tau))
        except Exception:
            continue
        rows.append(
            {
                "ttm_years": float(tau),
                "n_quotes": int(len(g)),
                "rmse_iv": rmse,
                "a": params.a,
                "b": params.b,
                "rho": params.rho,
                "m": params.m,
                "sigma": params.sigma,
            }
        )
    return pd.DataFrame(rows)


def smooth_panel_svi(
    df: pd.DataFrame, rate: float = 0.0, div: float = 0.0, min_quotes: int = 5
) -> pd.DataFrame:
    """Replace each quote's ``iv`` with its SVI-fitted value, slice by slice.

    Slices with fewer than ``min_quotes`` points (or that fail to fit) are left
    untouched. Requires columns ``date, spot, strike, iv`` and a maturity column.
    """
    df = df.copy()
    if "ttm_years" not in df.columns:
        from ivsh.data.clean import time_to_maturity_years

        df["ttm_years"] = time_to_maturity_years(df)

    iv = df["iv"].to_numpy(dtype=float).copy()
    for (_, tau), g in df.groupby(["date", "ttm_years"]):
        if len(g) < min_quotes:
            continue
        s = float(g["spot"].iloc[0])
        fwd = s * np.exp((rate - div) * float(tau))
        k = np.log(g["strike"].to_numpy(dtype=float) / fwd)
        w = g["iv"].to_numpy(dtype=float) ** 2 * float(tau)
        try:
            params, _ = fit_svi_slice(k, w, float(tau))
        except Exception:
            continue
        iv[g.index.to_numpy()] = params.iv(k, float(tau))
    df["iv"] = iv
    return df
