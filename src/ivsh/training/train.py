"""Training loops for the learnable hedging policies.

Both the prototype hedger and the black-box MLP are memoryless maps from
standardised state features to bounded holdings, so the whole hedging path is an
explicit numpy expression. We optimise the cost-adjusted CVaR utility with
L-BFGS-B using **analytic gradients** (backprop through the policy and the P&L),
which is fast enough to train on many Monte-Carlo paths and to support
validation-based early stopping. The auxiliary CVaR variable ``eta`` is optimised
jointly with the policy parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ivsh.baselines.policies import delta_vega_hedge
from ivsh.envs.hedging_env import EpisodeBank
from ivsh.features.standardize import Standardizer
from ivsh.models.blackbox import MLPHedger
from ivsh.models.clustering import KMeansResult, kmeans
from ivsh.models.prototype_policy import ProtoSurfaceHedger
from ivsh.training.objective import cvar_from_pnl, neg_utility, softplus

_BETA = 50.0


@dataclass
class TrainConfig:
    cvar_alpha: float = 0.95
    cvar_weight: float = 1.0
    turnover_weight: float = 0.0
    max_iter: int = 400
    seed: int = 7
    l2: float = 1e-3  # weight decay on policy parameters (stabilises the MLP)
    # Anchor the policy as a bounded *residual* on top of the delta-vega Greek
    # hedge. Essential on real (non-martingale) data: it stops the policy from
    # speculating on in-sample drift and keeps it a genuine hedge.
    anchor: bool = False
    # prototype model
    n_prototypes: int = 8
    action_scale: float = 2.5
    # black-box model
    hidden: int = 16


def make_standardizer(train_bank: EpisodeBank) -> Standardizer:
    return Standardizer.fit(train_bank.flat_features())


def _policy_pnl(policy, bank: EpisodeBank, scaler: Standardizer, smooth: bool, base=None):
    x = scaler.transform(bank.flat_features())
    holdings = policy.predict_holdings(x).reshape(bank.n_episodes, bank.horizon, -1)
    if base is not None:
        holdings = holdings + base
    pnl = bank.episode_pnl(holdings, smooth_costs=smooth)
    return pnl, holdings


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def fit_policy(
    policy,
    train_bank: EpisodeBank,
    scaler: Standardizer,
    cfg: TrainConfig,
    val_bank: EpisodeBank | None = None,
):
    """Fit a policy (analytic gradient + L-BFGS-B, optional val early stopping)."""
    n = policy.n_params
    x_tr = scaler.transform(train_bank.flat_features())
    E, L = train_bank.n_episodes, train_bank.horizon
    alpha, lam, l2 = cfg.cvar_alpha, cfg.cvar_weight, cfg.l2
    theta0 = np.concatenate([policy.get_flat_params(), [0.0]])

    base_tr = delta_vega_hedge(train_bank) if cfg.anchor else None
    base_vl = delta_vega_hedge(val_bank) if (cfg.anchor and val_bank is not None and val_bank.n_episodes > 0) else None

    def obj_and_grad(z: np.ndarray):
        policy.set_flat_params(z[:n])
        eta = z[n]
        resid_flat, cache = policy.forward(x_tr)
        holdings = resid_flat.reshape(E, L, -1)
        if base_tr is not None:
            holdings = holdings + base_tr
        pnl, dpnl_dh = train_bank.pnl_grad(holdings, smooth_costs=True)
        loss = -pnl
        hinge = softplus(loss - eta, _BETA)
        cvar_term = eta + hinge.mean() / (1.0 - alpha)
        J = -pnl.mean() - lam * (-cvar_term) + l2 * float((z[:n] ** 2).sum())
        # gradients
        s = _sigmoid(_BETA * (loss - eta))  # softplus'
        c = (-1.0 / E) - (lam / ((1.0 - alpha) * E)) * s  # dJ/dpnl[e]
        grad_h = (c[:, None, None] * dpnl_dh).reshape(E * L, -1)
        grad_theta = policy.backward(grad_h, cache) + 2.0 * l2 * z[:n]
        grad_eta = lam * (1.0 - s.mean() / (1.0 - alpha))
        return float(J), np.concatenate([grad_theta, [grad_eta]])

    use_val = val_bank is not None and val_bank.n_episodes > 0
    best = {"val": np.inf, "z": theta0.copy()}

    def callback(z):
        if not use_val:
            return
        policy.set_flat_params(z[:n])
        pnl, _ = _policy_pnl(policy, val_bank, scaler, smooth=False, base=base_vl)
        val = cfg.cvar_weight * cvar_from_pnl(pnl, alpha) - pnl.mean()  # neg utility
        if val < best["val"]:
            best["val"] = val
            best["z"] = z.copy()

    res = minimize(
        obj_and_grad,
        theta0,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter": cfg.max_iter},
    )
    z_final = best["z"] if (use_val and np.isfinite(best["val"])) else res.x
    policy.set_flat_params(z_final[:n])
    pnl, _ = _policy_pnl(policy, train_bank, scaler, smooth=False, base=base_tr)
    return policy, {
        "success": bool(res.success),
        "n_iter": int(res.nit),
        "final_obj": float(res.fun),
        "train_mean_pnl": float(pnl.mean()),
        "train_cvar": float(cvar_from_pnl(pnl, alpha)),
    }


def fit_prototype(
    train_bank: EpisodeBank,
    scaler: Standardizer,
    cfg: TrainConfig,
    val_bank: EpisodeBank | None = None,
) -> tuple[ProtoSurfaceHedger, KMeansResult, dict]:
    x = scaler.transform(train_bank.flat_features())
    km = kmeans(x, cfg.n_prototypes, seed=cfg.seed)
    policy = ProtoSurfaceHedger(
        prototypes=km.centers,
        action_dim=2,
        action_scale=cfg.action_scale,
        seed=cfg.seed,
    )
    policy, history = fit_policy(policy, train_bank, scaler, cfg, val_bank=val_bank)
    return policy, km, history


def fit_blackbox(
    train_bank: EpisodeBank,
    scaler: Standardizer,
    cfg: TrainConfig,
    val_bank: EpisodeBank | None = None,
) -> tuple[MLPHedger, dict]:
    policy = MLPHedger(
        n_features=train_bank.n_features,
        hidden=cfg.hidden,
        action_dim=2,
        action_scale=cfg.action_scale,
        seed=cfg.seed,
    )
    policy, history = fit_policy(policy, train_bank, scaler, cfg, val_bank=val_bank)
    return policy, history
