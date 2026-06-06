from __future__ import annotations

import numpy as np

from ivsh.pricing.black_scholes import bs_greeks, bs_price, implied_vol


def test_put_call_parity():
    s, k, t, v, r, q = 100.0, 105.0, 0.5, 0.2, 0.03, 0.01
    call = bs_price(s, k, t, v, r, q, "call")
    put = bs_price(s, k, t, v, r, q, "put")
    lhs = call - put
    rhs = s * np.exp(-q * t) - k * np.exp(-r * t)
    assert abs(lhs - rhs) < 1e-8


def test_delta_bounds_and_vega_positive():
    g = bs_greeks(100.0, 100.0, 0.5, 0.2, option_type="call")
    assert 0.0 <= g["delta"] <= 1.0
    assert g["vega"] > 0
    assert g["gamma"] > 0
    gp = bs_greeks(100.0, 100.0, 0.5, 0.2, option_type="put")
    assert -1.0 <= gp["delta"] <= 0.0


def test_price_at_expiry_is_intrinsic():
    assert abs(bs_price(110.0, 100.0, 0.0, 0.2, option_type="call") - 10.0) < 1e-8
    assert abs(bs_price(90.0, 100.0, 0.0, 0.2, option_type="put") - 10.0) < 1e-8


def test_implied_vol_roundtrip():
    true_vol = 0.27
    price = bs_price(100.0, 95.0, 0.4, true_vol, 0.02, 0.0, "call")
    iv = implied_vol(price, 100.0, 95.0, 0.4, 0.02, 0.0, "call")
    assert abs(float(iv) - true_vol) < 1e-3


def test_vectorised_shapes():
    spot = np.array([90.0, 100.0, 110.0])
    p = bs_price(spot, 100.0, 0.3, 0.2, option_type="call")
    assert p.shape == (3,)
    assert np.all(np.diff(p) > 0)  # call price increases in spot
