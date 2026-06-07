"""The HedgingGymEnv reward stream must reproduce EpisodeBank.episode_pnl."""

import numpy as np
import pytest

from ivsh.data.market import MarketConfig, simulate_market
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank
from ivsh.baselines.policies import delta_vega_hedge
from ivsh.training.train import make_standardizer

gym = pytest.importorskip("gymnasium")
from ivsh.models.deep_rl import HedgingGymEnv  # noqa: E402


def _small_bank():
    market = simulate_market(MarketConfig(n_days=180, seed=3))
    return build_episode_bank(market, EnvConfig(episode_stride=10))


def test_env_return_matches_episode_pnl():
    bank = _small_bank()
    scaler = make_standardizer(bank)
    env = HedgingGymEnv(bank, scaler, action_scale=1.5, seed=0, random_reset=False)
    base = delta_vega_hedge(bank)

    # Pick a fixed residual per (episode, step) and check the gym return equals
    # episode_pnl for the equivalent holdings tensor.
    rng = np.random.default_rng(1)
    residual_unit = rng.uniform(-1, 1, size=(bank.n_episodes, bank.horizon, 2))
    holdings = base + residual_unit * env.action_scale

    pnl_ref = bank.episode_pnl(holdings)

    for e in range(min(bank.n_episodes, 12)):
        env._e = e  # deterministic episode selection
        obs, _ = env.reset()
        total = 0.0
        t = 0
        done = False
        while not done:
            _, r, done, _, _ = env.step(residual_unit[e, t])
            total += r
            t += 1
        assert t == bank.horizon
        assert total == pytest.approx(pnl_ref[e], rel=1e-9, abs=1e-9)


def test_residual_scale_zero_recovers_delta_vega():
    bank = _small_bank()
    scaler = make_standardizer(bank)
    env = HedgingGymEnv(bank, scaler, action_scale=1.5, residual_scale=0.0,
                        seed=0, random_reset=False)
    base = delta_vega_hedge(bank)
    pnl_dv = bank.episode_pnl(base)
    for e in range(min(bank.n_episodes, 6)):
        env._e = e
        env.reset()
        total = 0.0
        done = False
        t = 0
        while not done:
            _, r, done, _, _ = env.step(np.array([1.0, -1.0]))  # any action, scaled to 0
            total += r
            t += 1
        assert total == pytest.approx(pnl_dv[e], rel=1e-9, abs=1e-9)
