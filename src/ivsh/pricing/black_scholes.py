"""Vectorised Black-Scholes pricing and Greeks.

All functions accept scalars or broadcastable numpy arrays. Prices and Greeks are
expressed per one unit of the option on one unit of the underlying. A continuous
dividend / carry yield ``q`` is supported so the same code prices index options
with a forward ``F = S * exp((r - q) * tau)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorised

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _d1_d2(
    spot: np.ndarray,
    strike: np.ndarray,
    ttm: np.ndarray,
    vol: np.ndarray,
    rate: float,
    div: float,
):
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    ttm = np.asarray(ttm, dtype=float)
    vol = np.asarray(vol, dtype=float)

    # Guard against zero time / vol so Greeks stay finite at expiry.
    safe_ttm = np.maximum(ttm, 1e-8)
    safe_vol = np.maximum(vol, 1e-8)
    sqrt_t = np.sqrt(safe_ttm)
    d1 = (np.log(spot / strike) + (rate - div + 0.5 * safe_vol**2) * safe_ttm) / (
        safe_vol * sqrt_t
    )
    d2 = d1 - safe_vol * sqrt_t
    return d1, d2, safe_ttm, safe_vol, sqrt_t


def bs_price(
    spot,
    strike,
    ttm,
    vol,
    rate: float = 0.0,
    div: float = 0.0,
    option_type: str = "call",
) -> np.ndarray:
    """Black-Scholes price. ``option_type`` is ``"call"`` or ``"put"``."""
    d1, d2, ttm_s, _, _ = _d1_d2(spot, strike, ttm, vol, rate, div)
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    disc_r = np.exp(-rate * ttm_s)
    disc_q = np.exp(-div * ttm_s)
    if option_type == "call":
        price = spot * disc_q * ndtr(d1) - strike * disc_r * ndtr(d2)
    elif option_type == "put":
        price = strike * disc_r * ndtr(-d2) - spot * disc_q * ndtr(-d1)
    else:
        raise ValueError(f"unknown option_type: {option_type!r}")
    # At/after expiry collapse to intrinsic value.
    intrinsic = (
        np.maximum(spot - strike, 0.0)
        if option_type == "call"
        else np.maximum(strike - spot, 0.0)
    )
    return np.where(np.asarray(ttm, dtype=float) <= 1e-8, intrinsic, price)


def bs_greeks(
    spot,
    strike,
    ttm,
    vol,
    rate: float = 0.0,
    div: float = 0.0,
    option_type: str = "call",
) -> dict[str, np.ndarray]:
    """Return delta, gamma, vega, theta, vanna, volga (first/second order).

    Vega/vanna/volga are reported per unit change in volatility (not per 1%),
    i.e. d/d(sigma). Theta is per year.
    """
    d1, d2, ttm_s, vol_s, sqrt_t = _d1_d2(spot, strike, ttm, vol, rate, div)
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    pdf_d1 = _norm_pdf(d1)
    disc_q = np.exp(-div * ttm_s)
    disc_r = np.exp(-rate * ttm_s)

    if option_type == "call":
        delta = disc_q * ndtr(d1)
    elif option_type == "put":
        delta = disc_q * (ndtr(d1) - 1.0)
    else:
        raise ValueError(f"unknown option_type: {option_type!r}")

    gamma = disc_q * pdf_d1 / (spot * vol_s * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t
    vanna = -disc_q * pdf_d1 * d2 / vol_s
    volga = vega * d1 * d2 / vol_s

    # Theta (per year), call/put.
    term1 = -spot * disc_q * pdf_d1 * vol_s / (2.0 * sqrt_t)
    if option_type == "call":
        theta = (
            term1
            - rate * strike * disc_r * ndtr(d2)
            + div * spot * disc_q * ndtr(d1)
        )
    else:
        theta = (
            term1
            + rate * strike * disc_r * ndtr(-d2)
            - div * spot * disc_q * ndtr(-d1)
        )

    expired = np.asarray(ttm, dtype=float) <= 1e-8
    zero = np.zeros_like(np.asarray(spot, dtype=float) + np.asarray(strike, dtype=float))
    return {
        "delta": np.where(expired, zero, delta),
        "gamma": np.where(expired, zero, gamma),
        "vega": np.where(expired, zero, vega),
        "theta": np.where(expired, zero, theta),
        "vanna": np.where(expired, zero, vanna),
        "volga": np.where(expired, zero, volga),
    }


def implied_vol(
    price,
    spot,
    strike,
    ttm,
    rate: float = 0.0,
    div: float = 0.0,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Recover implied vol by bisection (robust, vectorised)."""
    price = np.asarray(price, dtype=float)
    lo = np.full_like(price, 1e-4)
    hi = np.full_like(price, 5.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = bs_price(spot, strike, ttm, mid, rate, div, option_type)
        too_high = val > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
        if np.all(hi - lo < tol):
            break
    return 0.5 * (lo + hi)
