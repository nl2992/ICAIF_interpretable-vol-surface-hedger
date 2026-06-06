from __future__ import annotations

import numpy as np

from ivsh.data.market import MarketConfig, simulate_market
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank

MARKET = simulate_market(MarketConfig(n_days=200, seed=3))
BANK = build_episode_bank(MARKET, EnvConfig())


def test_no_trade_pnl_identity():
    # Unhedged P&L must equal premium received minus terminal payoff.
    holdings = np.zeros((BANK.n_episodes, BANK.horizon, 2))
    pnl = BANK.episode_pnl(holdings)
    premium = BANK.premium
    payoff = BANK.config.notional * BANK.v_liab[:, -1]
    assert np.allclose(pnl, premium - payoff, atol=1e-8)


def test_zero_cost_hedge_sanity():
    cfg = EnvConfig(underlying_cost_bps=0.0, option_cost_bps=0.0)
    bank = build_episode_bank(MARKET, cfg)
    holdings = np.zeros((bank.n_episodes, bank.horizon, 2))
    holdings[:, :, 0] = 0.5
    # With zero costs, a constant share position adds q * (S_T - S_0) telescoped.
    pnl = bank.episode_pnl(holdings)
    expected_hedge = 0.5 * (bank.spot[:, -1] - bank.spot[:, 0])
    liab = -cfg.notional * (bank.v_liab[:, -1] - bank.v_liab[:, 0])
    assert np.allclose(pnl, liab + expected_hedge, atol=1e-8)


def test_costs_reduce_pnl():
    holdings = np.zeros((BANK.n_episodes, BANK.horizon, 2))
    holdings[:, :, 0] = np.linspace(0, 1, BANK.horizon)  # churns -> turnover
    free = build_episode_bank(MARKET, EnvConfig(underlying_cost_bps=0.0, option_cost_bps=0.0))
    pnl_cost = BANK.episode_pnl(holdings).sum()
    pnl_free = free.episode_pnl(holdings).sum()
    assert pnl_cost < pnl_free


def test_pnl_grad_matches_finite_difference():
    rng = np.random.default_rng(0)
    holdings = 0.3 * rng.standard_normal((BANK.n_episodes, BANK.horizon, 2))
    pnl, grad = BANK.pnl_grad(holdings, smooth_costs=True)
    eps = 1e-6
    # check a few random entries
    for _ in range(8):
        e = rng.integers(BANK.n_episodes)
        j = rng.integers(BANK.horizon)
        c = rng.integers(2)
        hp = holdings.copy(); hp[e, j, c] += eps
        hm = holdings.copy(); hm[e, j, c] -= eps
        fd = (BANK.episode_pnl(hp, smooth_costs=True)[e] - BANK.episode_pnl(hm, smooth_costs=True)[e]) / (2 * eps)
        assert abs(fd - grad[e, j, c]) < 1e-4
