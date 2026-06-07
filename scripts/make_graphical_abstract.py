"""Single seminal 'graphical abstract' that conveys the whole narrative:

  (1) STATE   -- the market is encoded as a volatility-surface regime;
  (2) DECISION-- the hedge is an interpretable, bounded residual on delta-vega,
                 built from a few named prototype regimes (the explanation IS the trade);
  (3) OUTCOME -- across two real markets the prototype matches delta-vega while
                 deep-RL/black-box hedgers blow up -- robust and auditable.

All panels are data-grounded (real SPY surface + the confirmed test CVaRs).

    python scripts/make_graphical_abstract.py
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ivsh.baselines.policies import delta_vega_hedge
from ivsh.data.market import TAU0
from ivsh.evaluation.backtest import run_baseline
from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer
from ivsh.utils.splits import chronological_split, subset

from ivsh.viz import METHOD_COLORS, apply_theme

ROOT = _pl.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGS = ROOT / "reports_real" / "figures"
apply_theme()

WINNER = dict(n_prototypes=8, l2=1e-3, action_scale=1.5, anchor=True,
              cvar_weight=3.0, cvar_alpha=0.975, residual_l2=0.0)


def _surface(params, moneyness, tenor_days):
    level, skew, curv, slope = params
    logm = np.log(moneyness)
    tau = tenor_days / 252.0
    iv = (level + skew * logm[None, :] + curv * logm[None, :] ** 2
          + slope * np.log(tau / TAU0)[:, None])
    return np.maximum(iv, 0.02)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "bank_spy.pkl", "rb") as f:
        bank = pickle.load(f)["bank"]
    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    scaler = make_standardizer(trb)
    proto, _, _ = fit_prototype(trb, scaler, TrainConfig(max_iter=300, seed=7, **WINNER), val_bank=vlb)

    # prototype surface params + a stressed test state's activation
    raw = proto.prototypes * scaler.std + scaler.mean
    params = raw[:, :4]
    stress_proto = int(np.argmax(params[:, 0]))
    dv_pnl = run_baseline(teb, "delta_vega")["pnl"]
    e = int(np.argmin(dv_pnl))  # worst delta-vega (stressed) episode
    xstate = scaler.transform(teb.flat_features()).reshape(teb.n_episodes, teb.horizon, -1)[e, 0]
    w = proto.weights(xstate[None, :])[0]  # [K]

    # outcome CVaRs (confirmed winner + the learned hedgers)
    conf = pd.read_csv(ROOT / "reports_real" / "tables" / "winner_confirmation.csv").set_index("universe")
    comp = pd.read_csv(ROOT / "reports_real" / "tables" / "multiverse_comparison.csv")
    L = {(r.universe, r.method): r.cvar_95 for r in comp.itertuples()}
    methods = ["delta_vega", "prototype", "blackbox", "ppo", "sac"]
    labels = ["delta-vega\n(classical)", "prototype\n(ours)", "MLP", "PPO", "SAC"]
    colors = [METHOD_COLORS[m] for m in methods]
    vals = {u: [conf.loc[u, "delta_vega_cvar"], conf.loc[u, "proto_cvar_mean"],
                L[(u, "blackbox")], L[(u, "ppo")], L[(u, "sac")]] for u in ["spy", "qqq"]}

    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.35], wspace=0.28,
                          left=0.02, right=0.985, top=0.82, bottom=0.12)

    # ---- Panel 1: STATE (3D surface) ----
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    mny = np.linspace(0.8, 1.2, 40); ten = np.array([7, 30, 60, 90, 180, 365])
    M, T = np.meshgrid(mny, ten)
    ax1.plot_surface(M, T, _surface(params[stress_proto], mny, ten), cmap="viridis",
                     edgecolor="k", linewidth=0.1, alpha=0.95)
    ax1.set_xlabel("moneyness", fontsize=8, labelpad=-4)
    ax1.set_ylabel("tenor (d)", fontsize=8, labelpad=-4)
    ax1.set_zlabel("impl. vol", fontsize=8, labelpad=-4)
    ax1.tick_params(labelsize=6, pad=-1)
    ax1.view_init(elev=24, azim=-125)
    ax1.set_title("(1) STATE\nencode the volatility-surface regime", fontsize=11)

    # ---- Panel 2: DECISION (interpretable, bounded) ----
    ax2 = fig.add_subplot(gs[0, 1])
    order = np.argsort(w)[::-1]
    bars = ax2.bar(range(len(w)), w[order], color="#8172b3", edgecolor="k", linewidth=0.5)
    ax2.set_xticks(range(len(w)))
    ax2.set_xticklabels([f"P{j}" for j in order], fontsize=8)
    ax2.set_ylabel("activation weight $w_k$", fontsize=9)
    ax2.set_title("(2) DECISION\ninterpretable & bounded", fontsize=11)
    ax2.text(0.5, -0.28, r"$\mathrm{hedge} = \Delta\nu\mathrm{\ hedge} + \sum_k w_k\, a_k$",
             transform=ax2.transAxes, ha="center", va="top", fontsize=12)
    ax2.text(0.5, -0.42, "anchor (classical)        +    bounded, vol-capped residual",
             transform=ax2.transAxes, ha="center", va="top", fontsize=8, color="#555")
    ax2.text(0.5, -0.56, "every trade traces to a few named regimes",
             transform=ax2.transAxes, ha="center", va="top", fontsize=8.5, style="italic", color="#444")

    # ---- Panel 3: OUTCOME (robust across markets) ----
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(2); bw = 0.16
    for i, (m, lab, c) in enumerate(zip(methods, labels, colors)):
        ax3.bar(x + (i - 2) * bw, [vals["spy"][i], vals["qqq"][i]], bw, label=lab,
                color=c, edgecolor="k", linewidth=0.4)
    ax3.set_yscale("log")
    ax3.set_ylim(1.0, 320)
    ax3.set_xticks(x); ax3.set_xticklabels(["SPY", "QQQ"])
    ax3.set_ylabel("test CVaR$_{95}$ (log)", fontsize=9)
    ax3.set_title("(3) OUTCOME\nrobust across two markets", fontsize=11)
    ax3.legend(fontsize=7.5, ncol=3, loc="upper center", framealpha=0.95,
               columnspacing=0.9, handlelength=1.2)
    # shade the "safe" band up to delta-vega level
    ax3.axhspan(1.0, max(vals["spy"][0], vals["qqq"][0]) * 1.05, color="#55a868", alpha=0.07)
    ax3.annotate("deep hedgers\nblow up (10-90x)", xy=(1 + bw, vals["qqq"][3]), xytext=(1.18, 200),
                 fontsize=8.5, color="#8c4b4b", ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#8c4b4b", lw=1.1))
    ax3.annotate("prototype $\\approx$ delta-vega\n(safe & auditable)", xy=(0 - bw, vals["spy"][1]),
                 xytext=(-0.02, 18), fontsize=8.5, color="#5a4a78", ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#5a4a78", lw=1.1))

    # ---- connecting arrows between panels (figure coords) ----
    for x0, x1 in [(0.345, 0.378), (0.66, 0.69)]:
        fig.add_artist(FancyArrowPatch((x0, 0.45), (x1, 0.45), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=22, lw=2, color="#333"))

    fig.suptitle("Interpretable volatility-surface hedging: matches classical hedging on the tail, "
                 "dominates deep hedgers, and is fully auditable", fontsize=13, y=0.97)
    fig.savefig(FIGS / "hero_graphical_abstract.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote hero_graphical_abstract.png")


if __name__ == "__main__":
    main()
