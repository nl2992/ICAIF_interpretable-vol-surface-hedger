from __future__ import annotations

import numpy as np
import pandas as pd


def cvar(returns: pd.Series | np.ndarray, alpha: float = 0.95) -> float:
    """Lower-tail CVaR reported as a positive loss number."""
    values = np.asarray(returns, dtype=float)
    if values.size == 0:
        raise ValueError("returns cannot be empty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    cutoff = np.quantile(values, 1 - alpha)
    tail = values[values <= cutoff]
    return float(-tail.mean())


def turnover(actions: np.ndarray) -> float:
    if actions.ndim != 2:
        raise ValueError("actions must be [time, action_dim]")
    return float(np.abs(np.diff(actions, axis=0)).sum())

