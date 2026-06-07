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
    # "do-no-harm" shrink-to-base: penalise the magnitude of the applied residual
    # (holdings - delta-vega base) so the policy defers to delta-vega unless a
    # deviation genuinely helps. 0.0 reproduces the unconstrained policy.
    residual_l2: float = 0.0


def make_standardizer(train_bank: EpisodeBank) -> Standardizer:
    return Standardizer.fit(train_bank.flat_features())


def realized_vol_scale(
    bank: EpisodeBank, ref: float | None = None, floor: float = 0.25
) -> tuple[np.ndarray, float]:
    """Causal volatility-scaled residual cap multiplier, shape ``[E, L]``.

    Shrinks the learned residual when trailing realised vol is high so that, in
    stress regimes, the anchored policy collapses toward the pure delta-vega
    hedge instead of speculating (the COVID-2020 walk-forward failure mode).

    ``scale = clip(ref / max(realised_vol, ref), floor, 1.0)`` — uses the raw
    ``realized_vol`` feature (already causal). ``ref`` defaults to the median
    realised vol over ``bank`` (pass the *train* ref when scaling val/test so the
    cap is consistent and leak-free).
    """
    idx = bank.feature_names.index("realized_vol")
    rv = bank.features[:, :, idx]  # [E, L] raw (unstandardised) realised vol
    if ref is None:
        ref = float(np.median(rv))
    ref = max(ref, 1e-6)
    scale = np.clip(ref / np.maximum(rv, ref), floor, 1.0)
    return scale, ref


def _apply_residual(residual: np.ndarray, scale, base):
    """Combine a raw residual ``[E, L, 2]`` with an optional ``[E, L]`` vol cap
    and an optional delta-vega ``base`` into final holdings."""
    if scale is not None:
        residual = residual * np.asarray(scale)[:, :, None]
    return residual + base if base is not None else residual


def _policy_pnl(policy, bank: EpisodeBank, scaler: Standardizer, smooth: bool, base=None, scale=None):
    x = scaler.transform(bank.flat_features())
    residual = policy.predict_holdings(x).reshape(bank.n_episodes, bank.horizon, -1)
    holdings = _apply_residual(residual, scale, base)
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
    residual_scale: np.ndarray | None = None,
    val_residual_scale: np.ndarray | None = None,
):
    """Fit a policy (analytic gradient + L-BFGS-B, optional val early stopping).

    ``residual_scale`` (``[E, L]``, optional) caps the learned residual per
    decision point — e.g. the volatility-scaled cap from
    :func:`realized_vol_scale`. ``None`` reproduces the unconstrained behaviour.
    """
    n = policy.n_params
    x_tr = scaler.transform(train_bank.flat_features())
    E, L = train_bank.n_episodes, train_bank.horizon
    alpha, lam, l2 = cfg.cvar_alpha, cfg.cvar_weight, cfg.l2
    rl2 = cfg.residual_l2
    theta0 = np.concatenate([policy.get_flat_params(), [0.0]])

    base_tr = delta_vega_hedge(train_bank) if cfg.anchor else None
    base_vl = delta_vega_hedge(val_bank) if (cfg.anchor and val_bank is not None and val_bank.n_episodes > 0) else None
    sc_tr = None if residual_scale is None else np.asarray(residual_scale)[:, :, None]

    def obj_and_grad(z: np.ndarray):
        policy.set_flat_params(z[:n])
        eta = z[n]
        resid_flat, cache = policy.forward(x_tr)
        residual = resid_flat.reshape(E, L, -1)
        holdings = residual * sc_tr if sc_tr is not None else residual
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
        grad_h = c[:, None, None] * dpnl_dh
        if rl2 > 0.0:  # shrink-to-base: penalise applied residual = holdings - base
            applied = holdings - base_tr if base_tr is not None else holdings
            J += rl2 * float((applied ** 2).sum()) / E
            grad_h = grad_h + (2.0 * rl2 / E) * applied
        if sc_tr is not None:  # chain rule: holdings = base + scale ⊙ residual
            grad_h = grad_h * sc_tr
        grad_theta = policy.backward(grad_h.reshape(E * L, -1), cache) + 2.0 * l2 * z[:n]
        grad_eta = lam * (1.0 - s.mean() / (1.0 - alpha))
        return float(J), np.concatenate([grad_theta, [grad_eta]])

    use_val = val_bank is not None and val_bank.n_episodes > 0
    best = {"val": np.inf, "z": theta0.copy()}

    def callback(z):
        if not use_val:
            return
        policy.set_flat_params(z[:n])
        pnl, _ = _policy_pnl(policy, val_bank, scaler, smooth=False, base=base_vl, scale=val_residual_scale)
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
    pnl, _ = _policy_pnl(policy, train_bank, scaler, smooth=False, base=base_tr, scale=residual_scale)
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
    residual_scale: np.ndarray | None = None,
    val_residual_scale: np.ndarray | None = None,
) -> tuple[ProtoSurfaceHedger, KMeansResult, dict]:
    x = scaler.transform(train_bank.flat_features())
    km = kmeans(x, cfg.n_prototypes, seed=cfg.seed)
    policy = ProtoSurfaceHedger(
        prototypes=km.centers,
        action_dim=2,
        action_scale=cfg.action_scale,
        seed=cfg.seed,
    )
    policy, history = fit_policy(
        policy, train_bank, scaler, cfg, val_bank=val_bank,
        residual_scale=residual_scale, val_residual_scale=val_residual_scale,
    )
    return policy, km, history


def fit_blackbox(
    train_bank: EpisodeBank,
    scaler: Standardizer,
    cfg: TrainConfig,
    val_bank: EpisodeBank | None = None,
    residual_scale: np.ndarray | None = None,
    val_residual_scale: np.ndarray | None = None,
) -> tuple[MLPHedger, dict]:
    policy = MLPHedger(
        n_features=train_bank.n_features,
        hidden=cfg.hidden,
        action_dim=2,
        action_scale=cfg.action_scale,
        seed=cfg.seed,
    )
    policy, history = fit_policy(
        policy, train_bank, scaler, cfg, val_bank=val_bank,
        residual_scale=residual_scale, val_residual_scale=val_residual_scale,
    )
    return policy, history
