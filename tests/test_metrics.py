from __future__ import annotations

import numpy as np

from ivsh.evaluation.metrics import compute_metrics, max_drawdown
from ivsh.evaluation.stats import paired_bootstrap_diff, wilcoxon_pnl
from ivsh.training.objective import cvar_from_pnl


def test_cvar_positive_for_losses():
    pnl = np.array([0.02, -0.01, -0.05, 0.01, -0.10])
    # tail of losses at alpha=0.6 -> worst 40%
    assert cvar_from_pnl(pnl, alpha=0.6) > 0


def test_metrics_keys_and_signs():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 500)
    m = compute_metrics(pnl, turnover=np.abs(rng.normal(0, 1, 500)))
    for k in ("mean_pnl", "cvar_95", "cvar_99", "worst", "max_drawdown", "turnover", "utility"):
        assert k in m
    assert m["cvar_99"] >= m["cvar_95"]  # deeper tail >= shallower tail


def test_max_drawdown_monotone_series():
    assert max_drawdown(np.array([1.0, 1.0, 1.0])) == 0.0
    assert max_drawdown(np.array([1.0, -2.0, 1.0])) == 2.0


def test_bootstrap_detects_better_tail():
    rng = np.random.default_rng(1)
    good = rng.normal(0, 1, 800)
    bad = rng.normal(0, 3, 800)  # fatter tail
    res = paired_bootstrap_diff(good, bad, stat="cvar", seed=1)
    assert res["diff"] < 0  # good has smaller tail loss
    assert res["ci_high"] < 0


def test_wilcoxon_runs():
    rng = np.random.default_rng(2)
    a = rng.normal(0.1, 1, 200)
    b = rng.normal(0, 1, 200)
    out = wilcoxon_pnl(a, b)
    assert 0.0 <= out["pvalue"] <= 1.0
