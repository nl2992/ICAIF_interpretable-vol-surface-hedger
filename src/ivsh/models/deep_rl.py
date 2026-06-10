"""Deep-RL hedging comparators (PPO / SAC via stable-baselines3).

The paper's original "black box" is a small numpy MLP trained by L-BFGS on the
analytic CVaR utility. This module adds a *stronger* deep-hedging comparator: a
proper policy-gradient / actor-critic agent (PPO or SAC) trained by interacting
with the hedging environment, so the paper can claim robustness against a real
deep-RL hedger rather than only a shallow MLP.

Design notes
------------
* The agent is a memoryless map from the **same standardised state features** the
  other policies use to a 2-D continuous action ``(d_shares, d_hedge_units)`` in
  ``[-1, 1]^2``. The action is scaled by ``action_scale`` and added to the
  delta-vega base holdings -- exactly the ``anchor=True`` residual formulation
  used by :func:`ivsh.evaluation.backtest.run_policy`.
* The per-step reward is the term-by-term decomposition of
  :meth:`EpisodeBank.episode_pnl`, so an episode's undiscounted return equals the
  hedging P&L that every other policy is scored on. This identity is unit-tested
  (``tests/test_deep_rl.py``).
* :func:`evaluate_sb3` does not step the gym loop; it batch-predicts the residual
  for every decision point and calls ``bank.episode_pnl`` directly, returning a
  dict in the **same contract** as ``run_policy`` so it drops straight into the
  metrics / significance code.

PPO and SAC optimise *expected* return (= mean hedging P&L) -- the standard
"deep hedging as RL" objective, the same mean-seeking target the paper ascribes
to a black box. CVaR is reported at evaluation, never optimised, keeping the
comparison clean.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from ivsh.baselines.policies import delta_vega_hedge
from ivsh.envs.hedging_env import EpisodeBank
from ivsh.features.standardize import Standardizer


@dataclass
class RLConfig:
    algo: str = "ppo"  # "ppo" | "sac"
    total_timesteps: int = 100_000
    action_scale: float = 1.5  # matches the analytic residual scale on real data
    seed: int = 7
    learning_rate: float = 3e-4
    gamma: float = 1.0  # finite horizon, no economic discounting of P&L
    verbose: int = 0
    device: str = "auto"  # "cuda" | "cpu" | "auto" (SB3 picks)
    downside_kappa: float = 0.0  # >0 = CVaR-shaped reward (extra penalty on per-step losses)


def _resolve_scale(residual_scale, E: int, L: int) -> np.ndarray:
    """Broadcast a residual-cap spec to an ``[E, L, 1]`` multiplier."""
    if residual_scale is None:
        return np.ones((E, L, 1))
    arr = np.asarray(residual_scale, dtype=float)
    if arr.ndim == 0:
        return np.full((E, L, 1), float(arr))
    if arr.shape == (E, L):
        return arr[:, :, None]
    raise ValueError(f"residual_scale must be scalar or [E, L]; got {arr.shape}")


class HedgingGymEnv(gym.Env):
    """Gymnasium environment that replays hedging episodes from an ``EpisodeBank``.

    One gym episode = one hedging episode. The action is a residual on the
    delta-vega hedge; the reward stream sums to ``bank.episode_pnl(holdings)``.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        bank: EpisodeBank,
        scaler: Standardizer,
        action_scale: float = 1.5,
        residual_scale=None,
        seed: int = 7,
        random_reset: bool = True,
        downside_kappa: float = 0.0,
    ):
        super().__init__()
        self.bank = bank
        self.scaler = scaler
        self.action_scale = float(action_scale)
        self.random_reset = random_reset
        self.downside_kappa = float(downside_kappa)
        self._rng = np.random.default_rng(seed)

        E, L, F = bank.features.shape
        self.E, self.L, self.F = E, L, F

        # Precompute everything the reward needs (term-by-term episode_pnl).
        self.feats = scaler.transform(bank.flat_features()).reshape(E, L, F).astype(np.float32)
        self.base = delta_vega_hedge(bank)  # [E, L, 2]
        self.scale_vec = _resolve_scale(residual_scale, E, L)  # [E, L, 1]
        self.ds = np.diff(bank.spot, axis=1)  # [E, L]
        self.do = np.diff(bank.o_hedge, axis=1)
        self.dv = np.diff(bank.v_liab, axis=1)
        self.spot_dec = bank.spot[:, :-1]
        self.ohedge_dec = bank.o_hedge[:, :-1]
        self.spot_T = bank.spot[:, -1]
        self.ohedge_T = bank.o_hedge[:, -1]
        cfg = bank.config
        self.notional = cfg.notional
        self.cs = cfg.underlying_cost_bps / 1e4
        self.co = cfg.option_cost_bps / 1e4

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(F,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._e = 0
        self._t = 0
        self._prev_q = np.zeros(2)

    # -- gymnasium API ----------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._e = int(self._rng.integers(0, self.E)) if self.random_reset else (self._e % self.E)
        self._t = 0
        self._prev_q = np.zeros(2)
        return self.feats[self._e, 0].copy(), {}

    def step(self, action):
        e, t = self._e, self._t
        residual = np.clip(action, -1.0, 1.0) * self.action_scale * self.scale_vec[e, t]
        q = self.base[e, t] + residual  # target holdings over [t, t+1]
        q_s, q_o = float(q[0]), float(q[1])

        # Term-by-term episode_pnl increment for this step.
        liab = -self.notional * self.dv[e, t]
        hedge = q_s * self.ds[e, t] + q_o * self.do[e, t]
        trade_s = q_s - self._prev_q[0]
        trade_o = q_o - self._prev_q[1]
        cost = abs(trade_s) * self.spot_dec[e, t] * self.cs + abs(trade_o) * self.ohedge_dec[e, t] * self.co
        reward = liab + hedge - cost

        self._prev_q = np.array([q_s, q_o])
        self._t += 1
        terminated = self._t >= self.L
        if terminated:
            # Terminal liquidation back to flat at the terminal prices.
            liq = abs(q_s) * self.spot_T[e] * self.cs + abs(q_o) * self.ohedge_T[e] * self.co
            reward -= liq
            obs = self.feats[e, self.L - 1].copy()  # dummy terminal obs
        else:
            obs = self.feats[e, self._t].copy()
        if self.downside_kappa > 0.0 and reward < 0.0:
            reward += self.downside_kappa * reward  # CVaR-shaped: amplify losses
        return obs, float(reward), terminated, False, {}


def _make_model(env, cfg: RLConfig):
    if cfg.algo == "ppo":
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy", env, learning_rate=cfg.learning_rate, gamma=cfg.gamma,
            seed=cfg.seed, verbose=cfg.verbose, device=cfg.device,
        )
    if cfg.algo == "sac":
        from stable_baselines3 import SAC

        return SAC(
            "MlpPolicy", env, learning_rate=cfg.learning_rate, gamma=cfg.gamma,
            seed=cfg.seed, verbose=cfg.verbose, device=cfg.device,
        )
    raise ValueError(f"unknown algo {cfg.algo!r}; use 'ppo' or 'sac'")


def train_sb3(
    train_bank: EpisodeBank,
    scaler: Standardizer,
    cfg: RLConfig,
    residual_scale=None,
):
    """Train a PPO/SAC hedger on ``train_bank``; returns the SB3 model."""
    env = HedgingGymEnv(
        train_bank, scaler, action_scale=cfg.action_scale,
        residual_scale=residual_scale, seed=cfg.seed, random_reset=True,
        downside_kappa=cfg.downside_kappa,
    )
    model = _make_model(env, cfg)
    model.learn(total_timesteps=cfg.total_timesteps, progress_bar=False)
    return model


def evaluate_sb3(
    model,
    bank: EpisodeBank,
    scaler: Standardizer,
    action_scale: float = 1.5,
    residual_scale=None,
):
    """Deterministic rollout -> dict matching ``run_policy`` (pnl/turnover/holdings).

    Batch-predicts the residual for every decision point and reuses
    ``bank.episode_pnl``, so it is identical-by-construction to how every other
    policy is scored.
    """
    E, L, F = bank.features.shape
    x = scaler.transform(bank.flat_features()).astype(np.float32)
    actions, _ = model.predict(x, deterministic=True)
    actions = np.clip(np.asarray(actions, dtype=float), -1.0, 1.0)
    scale_vec = _resolve_scale(residual_scale, E, L)
    residual = actions.reshape(E, L, 2) * action_scale * scale_vec
    holdings = residual + delta_vega_hedge(bank)
    return {
        "pnl": bank.episode_pnl(holdings),
        "turnover": bank.turnover(holdings),
        "holdings": holdings,
    }
