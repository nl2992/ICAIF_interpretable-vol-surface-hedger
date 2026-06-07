"""Tests for the H4/H6/H7 grid-search mechanisms."""

import numpy as np
import pytest

from ivsh.data.market import MarketConfig, simulate_market
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank
from ivsh.baselines.policies import delta_vega_hedge
from ivsh.evaluation.backtest import run_policy, run_policy_ensemble
from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer
from ivsh.utils.splits import stress_resample_index, subset


def _bank(seed=3, n_days=260):
    return build_episode_bank(simulate_market(MarketConfig(n_days=n_days, seed=seed)),
                              EnvConfig(episode_stride=8))


# ---- H4: shrink-to-base residual penalty -------------------------------------
def test_residual_l2_shrinks_toward_delta_vega():
    bank = _bank()
    sc = make_standardizer(bank)
    base_pnl = bank.episode_pnl(delta_vega_hedge(bank))
    cfg0 = TrainConfig(n_prototypes=6, max_iter=80, anchor=True, action_scale=1.5, seed=7)
    cfg_big = TrainConfig(n_prototypes=6, max_iter=80, anchor=True, action_scale=1.5,
                          seed=7, residual_l2=1e3)
    p0, _, _ = fit_prototype(bank, sc, cfg0)
    pb, _, _ = fit_prototype(bank, sc, cfg_big)
    res0 = run_policy(p0, bank, sc, anchor=True)["holdings"] - delta_vega_hedge(bank)
    resb = run_policy(pb, bank, sc, anchor=True)["holdings"] - delta_vega_hedge(bank)
    # A large penalty must drive the applied residual toward zero.
    assert np.abs(resb).mean() < 0.1 * np.abs(res0).mean() + 1e-9
    # ...and the policy P&L toward the delta-vega base P&L.
    pnl_b = run_policy(pb, bank, sc, anchor=True)["pnl"]
    assert np.abs(pnl_b - base_pnl).mean() < 0.05


def test_residual_l2_gradient_matches_finite_difference():
    """The analytic objective gradient (with residual_l2) must match numerics."""
    from ivsh.models.prototype_policy import ProtoSurfaceHedger
    from ivsh.models.clustering import kmeans
    from ivsh.training.objective import cvar_from_pnl, softplus

    bank = _bank(seed=5, n_days=200)
    sc = make_standardizer(bank)
    x = sc.transform(bank.flat_features())
    E, L = bank.n_episodes, bank.horizon
    base = delta_vega_hedge(bank)
    km = kmeans(x, 5, seed=1)
    policy = ProtoSurfaceHedger(km.centers, action_dim=2, action_scale=1.5, seed=1)
    n = policy.n_params
    rng = np.random.default_rng(0)
    z = np.concatenate([0.2 * rng.standard_normal(n), [0.3]])
    alpha, lam, rl2 = 0.95, 1.0, 5.0

    def objective(zz):
        policy.set_flat_params(zz[:n])
        eta = zz[n]
        res = policy.predict_holdings(x).reshape(E, L, -1)
        holdings = res + base
        pnl = bank.episode_pnl(holdings, smooth_costs=True)
        loss = -pnl
        cvar = eta + softplus(loss - eta, 50.0).mean() / (1 - alpha)
        applied = holdings - base
        return -pnl.mean() + lam * cvar + rl2 * (applied ** 2).sum() / E

    # analytic gradient via the same machinery fit_policy uses
    from ivsh.training import train as T
    policy.set_flat_params(z[:n])
    eta = z[n]
    res_flat, cache = policy.forward(x)
    holdings = res_flat.reshape(E, L, -1) + base
    pnl, dpnl_dh = bank.pnl_grad(holdings, smooth_costs=True)
    s = T._sigmoid(50.0 * ((-pnl) - eta))
    c = (-1.0 / E) - (lam / ((1 - alpha) * E)) * s
    grad_h = c[:, None, None] * dpnl_dh
    applied = holdings - base
    grad_h = grad_h + (2.0 * rl2 / E) * applied
    grad_theta = policy.backward(grad_h.reshape(E * L, -1), cache)

    # finite-difference check on a few param coordinates
    eps = 1e-6
    for i in rng.choice(n, size=6, replace=False):
        zp = z.copy(); zp[i] += eps
        zm = z.copy(); zm[i] -= eps
        num = (objective(zp) - objective(zm)) / (2 * eps)
        assert abs(num - grad_theta[i]) < 1e-3, (i, num, grad_theta[i])


# ---- H6: seed ensemble -------------------------------------------------------
def test_ensemble_single_policy_equals_run_policy():
    bank = _bank()
    sc = make_standardizer(bank)
    p, _, _ = fit_prototype(bank, sc, TrainConfig(n_prototypes=6, max_iter=60, anchor=True,
                                                  action_scale=1.5, seed=7))
    a = run_policy(p, bank, sc, anchor=True)["pnl"]
    b = run_policy_ensemble([p], bank, sc, anchor=True)["pnl"]
    assert np.allclose(a, b, atol=1e-9)


# ---- H7: stress resampling ---------------------------------------------------
def test_stress_resample_oversamples_stress():
    bank = _bank()
    idx = stress_resample_index(bank, power=3.0, seed=1)
    assert idx.shape[0] == bank.n_episodes
    assert idx.min() >= 0 and idx.max() < bank.n_episodes
    # higher power -> higher mean stress fraction than uniform
    uniform = bank.regime_frac_stress.mean()
    weighted = bank.regime_frac_stress[idx].mean()
    assert weighted >= uniform - 1e-9
