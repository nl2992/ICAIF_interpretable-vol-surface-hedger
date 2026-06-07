"""Run policies over an episode bank and collect P&L / diagnostics."""

from __future__ import annotations

import numpy as np

from ivsh.baselines.policies import BASELINES
from ivsh.envs.hedging_env import EpisodeBank
from ivsh.evaluation.metrics import compute_metrics
from ivsh.features.standardize import Standardizer

# Feature groups for ablations.
SURFACE_FEATURES = (
    "surf_level",
    "surf_skew",
    "surf_curv",
    "surf_slope",
    "atm_iv_short",
    "atm_iv_long",
    "term_slope",
    "realized_vol",
    "ret_5d",
    "dlevel_1d",
)
GREEK_FEATURES = (
    "liab_delta",
    "liab_vega",
    "liab_gamma",
    "liab_logmoney",
    "liab_ttm",
    "hedge_delta",
    "hedge_vega",
)


def run_baseline(bank: EpisodeBank, name: str):
    holdings = BASELINES[name](bank)
    return {
        "pnl": bank.episode_pnl(holdings),
        "turnover": bank.turnover(holdings),
        "holdings": holdings,
    }


def run_policy(policy, bank: EpisodeBank, scaler: Standardizer, anchor: bool = False,
               residual_scale: np.ndarray | None = None):
    """Evaluate a learned policy. If ``anchor`` the policy output is a residual
    added to the delta-vega Greek hedge. ``residual_scale`` (``[E, L]``, optional)
    caps the residual per decision point (e.g. the volatility-scaled cap), and
    must match what was used at training time."""
    x = scaler.transform(bank.flat_features())
    holdings = policy.predict_holdings(x).reshape(bank.n_episodes, bank.horizon, -1)
    if residual_scale is not None:
        holdings = holdings * np.asarray(residual_scale)[:, :, None]
    if anchor:
        holdings = holdings + BASELINES["delta_vega"](bank)
    return {
        "pnl": bank.episode_pnl(holdings),
        "turnover": bank.turnover(holdings),
        "holdings": holdings,
    }


def run_policy_ensemble(policies, bank: EpisodeBank, scaler: Standardizer,
                        anchor: bool = False, residual_scale: np.ndarray | None = None):
    """Average the (capped) residuals of several policies into one hedge.

    Each policy contributes its standardised-state residual; we average them
    (variance reduction across seeds), apply the optional residual cap, and add
    the delta-vega base once. Returns the same dict contract as ``run_policy``.
    """
    x = scaler.transform(bank.flat_features())
    res = np.mean([p.predict_holdings(x).reshape(bank.n_episodes, bank.horizon, -1)
                   for p in policies], axis=0)
    if residual_scale is not None:
        res = res * np.asarray(residual_scale)[:, :, None]
    holdings = res + BASELINES["delta_vega"](bank) if anchor else res
    return {"pnl": bank.episode_pnl(holdings), "turnover": bank.turnover(holdings),
            "holdings": holdings}


def regime_metrics(pnl: np.ndarray, regime_start: np.ndarray) -> dict[str, dict]:
    out = {}
    for label, mask in (("calm", regime_start == 0), ("stress", regime_start == 1)):
        if mask.sum() > 1:
            out[label] = compute_metrics(pnl[mask])
            out[label]["n"] = int(mask.sum())
    return out
