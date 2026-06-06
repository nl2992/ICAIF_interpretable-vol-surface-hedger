from __future__ import annotations

import numpy as np
import pandas as pd

from ivsh.data.build_surface import build_surface_tensor, save_surface_tensor, surface_quality
from ivsh.data.loaders import market_from_option_panel
from ivsh.data.market import MarketConfig, simulate_market
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank
from ivsh.features.svi import SVIParams, fit_svi_slice, smooth_panel_svi


def test_fit_svi_recovers_params():
    true = SVIParams(a=0.02, b=0.10, rho=-0.4, m=0.0, sigma=0.15)
    k = np.linspace(-0.4, 0.4, 21)
    tau = 0.5
    w = true.total_variance(k)
    params, rmse = fit_svi_slice(k, w, tau)
    assert rmse < 1e-4
    assert np.allclose(params.total_variance(k), w, atol=1e-5)


def test_svi_iv_positive():
    p = SVIParams(0.04, 0.2, -0.3, 0.0, 0.2)
    assert np.all(p.iv(np.linspace(-0.5, 0.5, 11), 0.5) > 0)


def test_smooth_panel_svi_denoises():
    rng = np.random.default_rng(0)
    true = SVIParams(a=0.03, b=0.12, rho=-0.5, m=0.0, sigma=0.18)
    tau = 0.5
    strikes = np.linspace(80, 120, 21)
    k = np.log(strikes / 100.0)
    iv_true = true.iv(k, tau)
    iv_noisy = iv_true + rng.normal(0, 0.01, len(k))
    df = pd.DataFrame(
        {"date": 0, "spot": 100.0, "strike": strikes, "ttm_years": tau, "iv": iv_noisy}
    )
    out = smooth_panel_svi(df)
    err_noisy = np.sqrt(np.mean((iv_noisy - iv_true) ** 2))
    err_svi = np.sqrt(np.mean((out["iv"].to_numpy() - iv_true) ** 2))
    assert err_svi < err_noisy


def _panel_from_market(mkt, tenors=(7, 30, 90, 180), moneyness=tuple(np.round(np.linspace(0.8, 1.2, 9), 3))):
    rows = []
    for d in range(mkt.n_days):
        spot = float(mkt.spot[d])
        for tnr in tenors:
            for m in moneyness:
                strike = m * spot
                rows.append({"date": d, "spot": spot, "strike": strike,
                             "ttm_years": tnr / 252.0, "iv": float(mkt.iv(d, strike, d + tnr))})
    return pd.DataFrame(rows)


def test_loader_svi_method_builds_market_and_bank():
    mkt = simulate_market(MarketConfig(n_days=60, seed=1))
    panel = _panel_from_market(mkt)
    rec = market_from_option_panel(panel, rate=mkt.rate, div=mkt.div, surface_method="svi")
    assert rec.n_days == mkt.n_days
    assert np.all(np.isfinite(rec.level)) and np.all(rec.level > 0)
    bank = build_episode_bank(rec, EnvConfig())
    assert bank.n_episodes > 0


def test_surface_tensor_build_and_save(tmp_path):
    mkt = simulate_market(MarketConfig(n_days=20, seed=2))
    tensor, coords = build_surface_tensor(mkt)
    assert tensor.shape == (20, len(mkt.config.grid_tenor_days), len(mkt.config.grid_moneyness))
    assert np.all(tensor > 0)
    path = save_surface_tensor(tensor, coords, str(tmp_path / "proc" / "surface_tensor.npz"))
    loaded = np.load(path)
    assert loaded["iv"].shape == tensor.shape


def test_surface_quality_low_residual_on_self():
    mkt = simulate_market(MarketConfig(n_days=15, seed=3))
    panel = _panel_from_market(mkt)
    q = surface_quality(mkt, panel)
    assert q["rmse"] < 1e-2  # market.iv reproduces its own quotes closely
