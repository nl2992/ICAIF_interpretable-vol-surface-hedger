"""Deterministic Greek-based hedge baselines.

Each function maps an :class:`~ivsh.envs.hedging_env.EpisodeBank` to a holdings
tensor ``[E, L, 2]`` (underlying shares, hedge-option units). We are short
``notional`` liability options; holdings are chosen to neutralise the indicated
Greeks of the combined book.
"""

from __future__ import annotations

import numpy as np

from ivsh.envs.hedging_env import EpisodeBank

# Feature array column indices (must match FEATURE_NAMES in hedging_env.py).
_FEAT_ATM_IV_LONG = 5   # "atm_iv_long"  — 90-day ATM implied vol
_FEAT_LIAB_TTM = 14     # "liab_ttm"     — liability time-to-maturity (years)


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


def delta_gamma_hedge(bank: EpisodeBank) -> np.ndarray:
    """Delta-gamma hedge: size the hedge option to neutralise gamma, use underlying for delta.

    The hedge option's gamma is not stored directly, so we recover it from its
    vega using the Black-Scholes ATM relation  gamma = vega / (S * sigma * T).
    sigma is approximated by the 90-day ATM IV feature; T is liability TTM plus
    the fixed extra days carried by the longer-dated hedge option.
    """
    from ivsh.data.market import TRADING_DAYS

    n = bank.config.notional
    g = bank.greeks
    s = bank.spot[:, :-1]  # [E, L] — spot at decision points

    # Hedge option TTM ≈ liability TTM + fixed extra tenor
    extra_ttm = (bank.config.hedge_tenor_days - bank.config.liab_tenor_days) / TRADING_DAYS
    ttm_hedge = bank.features[:, :, _FEAT_LIAB_TTM] + extra_ttm      # [E, L]
    iv_hedge = bank.features[:, :, _FEAT_ATM_IV_LONG]                  # [E, L]

    # ATM BS: gamma = vega / (S * sigma * T)
    denom = s * np.maximum(iv_hedge, 1e-4) * np.maximum(ttm_hedge, 1e-4)
    gamma_hedge = g["vega_hedge"] / np.maximum(denom, 1e-8)

    q_o = n * g["gamma_liab"] / np.where(np.abs(gamma_hedge) < 1e-8, np.nan, gamma_hedge)
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
    "delta_gamma_vega": delta_gamma_hedge,
}
