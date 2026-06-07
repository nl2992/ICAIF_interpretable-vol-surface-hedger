"""Publication-grade "hero" visuals that carry the paper's story.

Four figures, all from real cached banks + the grid winner:
  1. surface_vocabulary  -- the model's learned regime "vocabulary" as actual 3D
     IV surfaces (a calm vs a stressed prototype), i.e. *what we contribute*.
  2. regime_map          -- 2D PCA embedding of market states coloured by regime,
     with prototype medoids and annotated crisis events.
  3. robustness_landscape-- CVaR across methods x {SPY,QQQ} x crisis folds; deep
     hedgers spike, the (winning) prototype stays flat.
  4. hedge_anatomy       -- one stressed episode end to end: reconstructed surface,
     prototype activation weights, residual on delta-vega, cumulative P&L.

    python scripts/make_hero_figures.py --config winner
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ivsh.baselines.policies import delta_vega_hedge
from ivsh.data.market import TAU0
from ivsh.evaluation.backtest import run_baseline, run_policy
from ivsh.training.objective import cvar_from_pnl
from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer, realized_vol_scale
from ivsh.utils.splits import chronological_split, subset

from ivsh.viz import METHOD_COLORS as COLORS, apply_theme

ROOT = _pl.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGS = ROOT / "reports_real" / "figures"
apply_theme()


def _load(uni):
    with open(ARTIFACTS / f"bank_{uni}.pkl", "rb") as f:
        return pickle.load(f)["bank"]


def _winner_cfg():
    """Read the grid winner's knobs, else a sensible regularised default."""
    w = ROOT / "reports_real" / "grid" / "grid_winner.csv"
    d = dict(action_scale=1.0, l2=1e-3, n_prototypes=8, cvar_weight=1.0,
             cvar_alpha=0.95, residual_l2=10.0, vol_floor=0.25)
    if w.exists():
        row = pd.read_csv(w).iloc[0]
        for k in d:
            if k in row and not pd.isna(row[k]):
                d[k] = type(d[k])(row[k]) if k != "vol_floor" else (
                    None if str(row[k]) in ("nan", "None") else float(row[k]))
    return d


def _fit_winner(bank):
    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    scaler = make_standardizer(trb)
    d = _winner_cfg()
    vf = d["vol_floor"]
    sc_tr = sc_te = None
    if vf is not None:
        sc_tr, ref = realized_vol_scale(trb, floor=vf)
        sc_te = realized_vol_scale(teb, ref=ref, floor=vf)[0]
    tc = TrainConfig(n_prototypes=int(d["n_prototypes"]), l2=d["l2"], max_iter=300,
                     anchor=True, action_scale=d["action_scale"], cvar_weight=d["cvar_weight"],
                     cvar_alpha=d["cvar_alpha"], residual_l2=d["residual_l2"], seed=7)
    proto, _, _ = fit_prototype(trb, scaler, tc, val_bank=vlb, residual_scale=sc_tr)
    return proto, scaler, teb, sc_te


def _surface(params, moneyness, tenor_days):
    level, skew, curv, slope = params
    logm = np.log(moneyness)
    tau = tenor_days / 252.0
    iv = (level + skew * logm[None, :] + curv * logm[None, :] ** 2
          + slope * np.log(tau / TAU0)[:, None])
    return np.maximum(iv, 0.02)


# 1 --------------------------------------------------------------------------- #
def fig_surface_vocabulary(proto, scaler):
    raw = proto.prototypes * scaler.std + scaler.mean
    params = raw[:, :4]
    levels = params[:, 0]
    calm, stress = int(np.argmin(levels)), int(np.argmax(levels))
    moneyness = np.linspace(0.8, 1.2, 40)
    tenor = np.array([7, 30, 60, 90, 180, 365])
    M, T = np.meshgrid(moneyness, tenor)
    fig = plt.figure(figsize=(11, 4.6))
    for i, (j, name, cmap) in enumerate([(calm, "calm-regime prototype", "Blues"),
                                          (stress, "stressed-regime prototype", "Reds")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        Z = _surface(params[j], moneyness, tenor)
        ax.plot_surface(M, T, Z, cmap=cmap, edgecolor="k", linewidth=0.15, alpha=0.95)
        ax.set_xlabel("moneyness K/F"); ax.set_ylabel("tenor (days)")
        ax.set_zlabel("implied vol")
        ax.set_title(f"P{j}: {name}")
        ax.view_init(elev=22, azim=-128)
    fig.suptitle("The learned regime vocabulary: each prototype is a readable IV surface",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(FIGS / "hero_surface_vocabulary.png", bbox_inches="tight"); plt.close(fig)


# 2 --------------------------------------------------------------------------- #
def fig_regime_map(proto, scaler, teb):
    X = scaler.transform(teb.flat_features())
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    protoproj = (proto.prototypes - X.mean(0)) @ Vt[:2].T
    reg = np.repeat(teb.regime_start, teb.horizon)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for r, c, lab in [(0, "#4c72b0", "calm"), (1, "#c44e52", "stress")]:
        m = reg == r
        ax.scatter(proj[m, 0], proj[m, 1], s=6, c=c, alpha=0.25, label=lab, edgecolors="none")
    ax.scatter(protoproj[:, 0], protoproj[:, 1], s=320, c="gold", marker="*",
               edgecolors="k", linewidths=1.2, zorder=5, label="prototypes")
    for j, (px, py) in enumerate(protoproj):
        ax.annotate(f"P{j}", (px, py), fontsize=9, fontweight="bold", ha="center", va="center")
    ax.set_xlabel("state PC-1"); ax.set_ylabel("state PC-2")
    ax.set_title("Market-state regime map: prototypes anchor calm and stressed regions")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(FIGS / "hero_regime_map.png", bbox_inches="tight"); plt.close(fig)


# 3 --------------------------------------------------------------------------- #
def fig_robustness_landscape():
    wf = ROOT / "reports_real" / "tables" / "walkforward_cvar.csv"
    wfq = ROOT / "reports_real" / "tables" / "walkforward_cvar_qqq.csv"
    if not wf.exists():
        print("  (skip robustness_landscape: walkforward CSVs missing)"); return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, path, title in [(axes[0], wf, "SPY"), (axes[1], wfq, "QQQ")]:
        if not _pl.Path(path).exists():
            ax.axis("off"); continue
        df = pd.read_csv(path).set_index("test_year")
        for m in [c for c in ["delta_vega", "prototype", "prototype_capped", "blackbox", "ppo"] if c in df]:
            ax.plot(df.index, df[m], marker="o", label=m, color=COLORS.get(m.replace("_capped", "")))
        ax.set_yscale("log"); ax.set_title(f"{title}: walk-forward CVaR95 by year")
        ax.set_xlabel("test year")
    axes[0].set_ylabel("CVaR95 (log scale)"); axes[0].legend(fontsize=8)
    fig.suptitle("Robustness landscape: deep hedgers blow up in crises; the prototype stays bounded",
                 y=1.0, fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "hero_robustness_landscape.png", bbox_inches="tight"); plt.close(fig)


# 4 --------------------------------------------------------------------------- #
def fig_hedge_anatomy(proto, scaler, teb, sc_te):
    out = run_policy(proto, teb, scaler, anchor=True, residual_scale=sc_te)
    base = delta_vega_hedge(teb)
    pnl_dv = run_baseline(teb, "delta_vega")["pnl"]
    e = int(np.argmin(pnl_dv))  # the worst delta-vega episode (stress)
    x = scaler.transform(teb.flat_features()).reshape(teb.n_episodes, teb.horizon, -1)[e]
    w = proto.weights(x)  # [L, K]
    raw = proto.prototypes * scaler.std + scaler.mean
    wsurf = (w.mean(0) @ raw[:, :4])
    moneyness = np.linspace(0.8, 1.2, 40); tenor = np.array([7, 30, 60, 90, 180, 365])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    Z = _surface(wsurf, moneyness, tenor)
    im = axes[0, 0].imshow(Z, aspect="auto", origin="lower", cmap="viridis",
                           extent=[0.8, 1.2, 0, len(tenor) - 1])
    axes[0, 0].set_yticks(range(len(tenor))); axes[0, 0].set_yticklabels(tenor)
    axes[0, 0].set_title("(a) episode's surface (prototype-weighted)")
    axes[0, 0].set_xlabel("moneyness"); axes[0, 0].set_ylabel("tenor (d)")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8, label="IV")

    axes[0, 1].imshow(w.T, aspect="auto", origin="lower", cmap="magma")
    axes[0, 1].set_title("(b) prototype activation over the episode")
    axes[0, 1].set_xlabel("rebalance step"); axes[0, 1].set_ylabel("prototype")

    resid = (out["holdings"][e] - base[e])
    axes[1, 0].plot(base[e][:, 0], label="delta-vega base (shares)", color="#55a868")
    axes[1, 0].plot(out["holdings"][e][:, 0], label="prototype (shares)", color="#8172b3")
    axes[1, 0].plot(resid[:, 0], "--", label="residual", color="#c44e52")
    axes[1, 0].axhline(0, color="k", lw=0.6)
    axes[1, 0].set_title("(c) holdings: small residual on the delta-vega hedge")
    axes[1, 0].set_xlabel("rebalance step"); axes[1, 0].set_ylabel("underlying holding")
    axes[1, 0].legend(fontsize=8)

    cum_p = np.cumsum(-np.diff(teb.v_liab[e]) * teb.config.notional
                      + out["holdings"][e][:, 0] * np.diff(teb.spot[e])
                      + out["holdings"][e][:, 1] * np.diff(teb.o_hedge[e]))
    cum_d = np.cumsum(-np.diff(teb.v_liab[e]) * teb.config.notional
                      + base[e][:, 0] * np.diff(teb.spot[e])
                      + base[e][:, 1] * np.diff(teb.o_hedge[e]))
    axes[1, 1].plot(cum_d, label="delta-vega", color="#55a868")
    axes[1, 1].plot(cum_p, label="prototype", color="#8172b3")
    axes[1, 1].set_title("(d) cumulative P&L on this stressed episode")
    axes[1, 1].set_xlabel("rebalance step"); axes[1, 1].set_ylabel("cumulative P&L")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Anatomy of one stressed hedge: surface -> active prototypes -> bounded residual -> P&L",
                 fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(FIGS / "hero_hedge_anatomy.png", bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="spy")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    bank = _load(args.universe)
    proto, scaler, teb, sc_te = _fit_winner(bank)
    for name, fn in [("surface_vocabulary", lambda: fig_surface_vocabulary(proto, scaler)),
                     ("regime_map", lambda: fig_regime_map(proto, scaler, teb)),
                     ("robustness_landscape", fig_robustness_landscape),
                     ("hedge_anatomy", lambda: fig_hedge_anatomy(proto, scaler, teb, sc_te))]:
        try:
            fn(); print(f"  wrote hero_{name}.png")
        except Exception as ex:  # one figure failing must not kill the rest
            print(f"  FAILED {name}: {ex!r}")


if __name__ == "__main__":
    main()
