"""Paired statistical comparisons between two policies' per-episode P&L.

Because every policy is evaluated on the *same* episodes, comparisons are
naturally paired, which sharpens the tests considerably.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from ivsh.training.objective import cvar_from_pnl


def paired_bootstrap_diff(
    pnl_a: np.ndarray,
    pnl_b: np.ndarray,
    stat: str = "cvar",
    alpha: float = 0.95,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 7,
) -> dict[str, float]:
    """Bootstrap CI for ``stat(a) - stat(b)`` resampling episodes jointly.

    For ``stat="cvar"`` a *negative* difference means policy A has a smaller tail
    loss than policy B (A is better).
    """
    pnl_a = np.asarray(pnl_a, dtype=float)
    pnl_b = np.asarray(pnl_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(pnl_a)

    def metric(x: np.ndarray) -> float:
        if stat == "cvar":
            return cvar_from_pnl(x, alpha)
        if stat == "mean":
            return float(x.mean())
        if stat == "utility":
            return float(x.mean() - cvar_from_pnl(x, alpha))
        raise ValueError(stat)

    point = metric(pnl_a) - metric(pnl_b)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = metric(pnl_a[idx]) - metric(pnl_b[idx])
    lo = float(np.quantile(diffs, (1 - ci) / 2))
    hi = float(np.quantile(diffs, 1 - (1 - ci) / 2))
    return {"diff": float(point), "ci_low": lo, "ci_high": hi, "p_two_sided": float(2 * min((diffs > 0).mean(), (diffs < 0).mean()))}


def wilcoxon_pnl(pnl_a: np.ndarray, pnl_b: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test on paired per-episode P&L (A - B)."""
    diff = np.asarray(pnl_a) - np.asarray(pnl_b)
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "pvalue": 1.0}
    stat, p = wilcoxon(diff)
    return {"statistic": float(stat), "pvalue": float(p)}
