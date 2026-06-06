"""Deterministic Greek-based hedge baselines.

Each function maps an :class:`~ivsh.envs.hedging_env.EpisodeBank` to a holdings
tensor ``[E, L, 2]`` (underlying shares, hedge-option units). We are short
``notional`` liability options; holdings are chosen to neutralise the indicated
Greeks of the combined book.
"""

from __future__ import annotations

import numpy as np

from ivsh.envs.hedging_env import EpisodeBank


def unhedged(bank: EpisodeBank) -> np.ndarray:
    return np.zeros((bank.n_episodes, bank.horizon, 2))


def delta_hedge(bank: EpisodeBank) -> np.ndarray:
    n = bank.config.notional
    h = np.zeros((bank.n_episodes, bank.horizon, 2))
    h[:, :, 0] = n * bank.greeks["delta_liab"]
    return h


def delta_vega_hedge(bank: EpisodeBank) -> np.ndarray:
    n = bank.config.notional
    g = bank.greeks
    q_o = n * g["vega_liab"] / np.where(np.abs(g["vega_hedge"]) < 1e-8, np.nan, g["vega_hedge"])
    q_o = np.nan_to_num(q_o, nan=0.0)
    q_s = n * g["delta_liab"] - q_o * g["delta_hedge"]
    h = np.zeros((bank.n_episodes, bank.horizon, 2))
    h[:, :, 0] = q_s
    h[:, :, 1] = q_o
    return h


BASELINES = {
    "unhedged": unhedged,
    "delta": delta_hedge,
    "delta_vega": delta_vega_hedge,
}
