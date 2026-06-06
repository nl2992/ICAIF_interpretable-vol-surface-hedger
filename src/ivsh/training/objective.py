"""Risk objective for training hedging policies.

We maximise a cost-adjusted CVaR utility

    U = E[PnL] - lambda * CVaR_alpha(loss) - kappa * E[turnover]

where ``loss = -PnL``. CVaR is written in the smooth Rockafellar-Uryasev form

    CVaR_alpha(loss) = min_eta  eta + 1/(1-alpha) * E[(loss - eta)_+]

so it can be optimised jointly with the policy parameters by adding ``eta`` as a
free variable. The training objective returns the *negative* utility (to be
minimised) and uses a softplus relaxation of the hinge for smooth gradients.
"""

from __future__ import annotations

import numpy as np


def softplus(x: np.ndarray, beta: float = 50.0) -> np.ndarray:
    # Numerically stable softplus, sharp enough to approximate the hinge.
    z = beta * x
    return np.where(z > 30, x, np.log1p(np.exp(np.clip(z, -30, 30))) / beta)


def cvar_from_pnl(pnl: np.ndarray, alpha: float = 0.95) -> float:
    """Empirical CVaR of the loss tail, reported as a positive number."""
    loss = -np.asarray(pnl, dtype=float)
    var = np.quantile(loss, alpha)
    tail = loss[loss >= var]
    if tail.size == 0:
        return float(var)
    return float(tail.mean())


def neg_utility(
    pnl: np.ndarray,
    eta: float,
    alpha: float = 0.95,
    cvar_weight: float = 1.0,
    turnover: np.ndarray | None = None,
    turnover_weight: float = 0.0,
    smooth: bool = True,
) -> float:
    """Negative cost-adjusted CVaR utility (objective to minimise)."""
    loss = -pnl
    hinge = softplus(loss - eta) if smooth else np.maximum(loss - eta, 0.0)
    cvar_term = eta + hinge.mean() / (1.0 - alpha)
    util = pnl.mean() - cvar_weight * cvar_term
    if turnover is not None and turnover_weight > 0:
        util -= turnover_weight * turnover.mean()
    return -float(util)
