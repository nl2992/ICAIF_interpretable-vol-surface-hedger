"""Paired statistical comparisons between two policies' per-episode P&L.

Because every policy is evaluated on the *same* episodes, comparisons are
naturally paired, which sharpens the tests considerably.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, wilcoxon

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


def stouffer_combine(results: list[dict[str, float]]) -> dict[str, float]:
    """Combine per-universe paired-bootstrap results into one cross-universe test.

    Each element of ``results`` is the output of :func:`paired_bootstrap_diff`
    (needs ``diff`` and ``p_two_sided``). We convert each two-sided p-value to a
    signed z-score (sign from the direction of ``diff``; for ``stat="cvar"`` a
    negative diff means A has the smaller tail) and combine with equal weights
    (Stouffer's Z). This respects each market's own P&L scale rather than pooling
    raw P&L across universes.
    """
    zs = []
    for r in results:
        p = min(max(r["p_two_sided"], 1e-12), 1.0)
        sign = -1.0 if r["diff"] < 0 else 1.0  # negative diff (A better) -> negative z
        zs.append(sign * norm.isf(p / 2.0))  # |z| from two-sided p
    zs = np.asarray(zs, dtype=float)
    z_comb = float(zs.sum() / np.sqrt(len(zs)))
    return {
        "z": z_comb,
        "p_two_sided": float(2.0 * norm.sf(abs(z_comb))),
        "mean_diff": float(np.mean([r["diff"] for r in results])),
        "n_universes": len(results),
    }


def wilcoxon_pnl(pnl_a: np.ndarray, pnl_b: np.ndarray) -> dict[str, float]:
    """Wilcoxon signed-rank test on paired per-episode P&L (A - B)."""
    diff = np.asarray(pnl_a) - np.asarray(pnl_b)
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "pvalue": 1.0}
    stat, p = wilcoxon(diff)
    return {"statistic": float(stat), "pvalue": float(p)}
