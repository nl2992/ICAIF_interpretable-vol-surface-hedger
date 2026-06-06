"""Hedging-performance metrics computed from a per-episode P&L distribution."""

from __future__ import annotations

import numpy as np

from ivsh.training.objective import cvar_from_pnl


def max_drawdown(pnl_sorted: np.ndarray) -> float:
    """Max drawdown of cumulative P&L (episodes ordered by start day)."""
    cum = np.cumsum(pnl_sorted)
    running_max = np.maximum.accumulate(cum)
    return float(np.max(running_max - cum)) if cum.size else 0.0


def compute_metrics(
    pnl: np.ndarray,
    turnover: np.ndarray | None = None,
    cvar_alpha: float = 0.95,
    cvar_weight: float = 1.0,
) -> dict[str, float]:
    pnl = np.asarray(pnl, dtype=float)
    cvar95 = cvar_from_pnl(pnl, 0.95)
    cvar99 = cvar_from_pnl(pnl, 0.99)
    metrics = {
        "mean_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "std_pnl": float(pnl.std()),
        "var_95": float(-np.quantile(pnl, 0.05)),
        "cvar_95": cvar95,
        "cvar_99": cvar99,
        "worst": float(-pnl.min()),
        "max_drawdown": max_drawdown(pnl),
        "utility": float(pnl.mean() - cvar_weight * cvar_from_pnl(pnl, cvar_alpha)),
    }
    if turnover is not None:
        metrics["turnover"] = float(np.asarray(turnover).mean())
    return metrics
