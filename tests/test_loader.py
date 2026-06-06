from __future__ import annotations

import numpy as np
import pandas as pd

from ivsh.data.loaders import clean_quotes, market_from_option_panel
from ivsh.data.market import MarketConfig, simulate_market
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank


def _panel_from_market(mkt, tenors=(7, 30, 90, 180), moneyness=(0.85, 0.95, 1.0, 1.05, 1.15)):
    rows = []
    for d in range(mkt.n_days):
        spot = float(mkt.spot[d])
        for tnr in tenors:
            for m in moneyness:
                strike = m * spot
                rows.append(
                    {
                        "date": int(d),
                        "spot": spot,
                        "strike": strike,
                        "ttm_years": tnr / 252.0,
                        "iv": float(mkt.iv(d, strike, d + tnr)),
                    }
                )
    return pd.DataFrame(rows)


def test_loader_recovers_surface_factors():
    mkt = simulate_market(MarketConfig(n_days=60, seed=1))
    panel = _panel_from_market(mkt)
    rec = market_from_option_panel(panel, rate=mkt.rate, div=mkt.div)
    assert rec.n_days == mkt.n_days
    assert np.allclose(rec.spot, mkt.spot, atol=1e-8)
    # factors are recovered (data generated from the exact parametric model)
    assert np.allclose(rec.level, mkt.level, atol=1e-3)
    assert np.allclose(rec.skew, mkt.skew, atol=1e-3)
    assert np.allclose(rec.slope, mkt.slope, atol=1e-3)


def test_loaded_market_builds_a_bank():
    mkt = simulate_market(MarketConfig(n_days=120, seed=2))
    panel = _panel_from_market(mkt)
    rec = market_from_option_panel(panel, rate=mkt.rate, div=mkt.div)
    bank = build_episode_bank(rec, EnvConfig())
    assert bank.n_episodes > 0
    # unhedged P&L identity still holds on the loaded market
    holdings = np.zeros((bank.n_episodes, bank.horizon, 2))
    pnl = bank.episode_pnl(holdings)
    assert np.allclose(pnl, bank.premium - bank.config.notional * bank.v_liab[:, -1], atol=1e-8)


def test_clean_quotes_filters():
    df = pd.DataFrame(
        {
            "date": [0, 0, 0, 0],
            "spot": [100.0] * 4,
            "strike": [100.0, 100.0, 100.0, 100.0],
            "ttm_years": [0.1, 0.1, 0.1, -0.1],  # last is expired
            "bid": [1.0, -1.0, 2.0, 1.0],  # second negative
            "ask": [1.2, 1.0, 1.0, 1.2],  # third crossed (bid>ask)
        }
    )
    clean, summary = clean_quotes(df)
    assert len(clean) == 1  # only the first row survives
    assert "removed" in summary.table.columns
