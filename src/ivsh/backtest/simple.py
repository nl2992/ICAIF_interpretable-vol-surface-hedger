from __future__ import annotations

import numpy as np
import pandas as pd


def hedge_pnl(actions: np.ndarray, risk_moves: np.ndarray, cost_bps: float = 1.0) -> pd.Series:
    """Toy hedge P&L: action exposure times risk move minus turnover cost."""
    if actions.shape != risk_moves.shape:
        raise ValueError("actions and risk_moves must have the same shape")
    gross = (actions * risk_moves).sum(axis=1)
    turnover = np.abs(np.diff(actions, axis=0, prepend=actions[:1])).sum(axis=1)
    costs = turnover * cost_bps / 10_000.0
    return pd.Series(gross - costs, name="hedge_pnl")

