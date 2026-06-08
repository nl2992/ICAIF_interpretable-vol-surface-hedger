"""Hedge anatomy figures for a stress episode (2020) and a calm episode (2019).

For each episode we show:
  (a) The input surface reconstructed from the prototype-weighted blend
  (b) Prototype activation weights over the episode
  (c) Holdings decomposition: delta-vega base, prototype residual, final position
  (d) Cumulative P&L: delta-vega vs prototype hedger

Example:
    python scripts/make_trade_anatomy.py --universe spy
    python scripts/make_trade_anatomy.py --universe spy --stress-year 2020 --calm-year 2019
"""

from __future__ import annotations

import argparse
import pathlib as _pl
import pickle
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ivsh.baselines.policies import delta_vega_hedge
from ivsh.data.market import TAU0
from ivsh.evaluation.backtest import run_baseline, run_policy
from ivsh.training.objective import cvar_from_pnl
from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer, realized_vol_scale
from ivsh.utils.splits import chronological_split, subset

from ivsh.viz import METHOD_COLORS as COLORS, apply_theme

apply_theme()

ROOT = _pl.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def _load(universe: str):
    with open(ARTIFACTS / f"bank_{universe}.pkl", "rb") as f:
        return pickle.load(f)


def _winner_cfg():
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
    return proto, scaler, teb, sc_te, sp


def _surface(params, moneyness, tenor_days):
    level, skew, curv, slope = params
    logm = np.log(moneyness)
    tau = tenor_days / 252.0
    iv = (level + skew * logm[None, :] + curv * logm[None, :] ** 2
          + slope * np.log(tau / TAU0)[:, None])
    return np.maximum(iv, 0.02)


def _pick_episode(teb, years_test, target_year: int, mode: str) -> int:
    """Return an episode index matching target_year and either 'stress' or 'calm'."""
    mask = years_test == target_year
    if not mask.any():
        # Fall back to closest year with episodes.
        avail = np.unique(years_test)
        target_year = avail[np.argmin(np.abs(avail - target_year))]
        mask = years_test == target_year
    pnl_dv = run_baseline(teb, "delta_vega")["pnl"]
    idxs = np.where(mask)[0]
    if mode == "stress":
        return int(idxs[np.argmin(pnl_dv[idxs])])   # worst P&L episode that year
    else:
        med = np.median(pnl_dv[idxs])
        return int(idxs[np.argmin(np.abs(pnl_dv[idxs] - med))])  # most typical episode


def _anatomy_figure(proto, scaler, teb, sc_te, episode_idx: int,
                    title: str, outpath: _pl.Path) -> None:
    out = run_policy(proto, teb, scaler, anchor=True, residual_scale=sc_te)
    base_h = delta_vega_hedge(teb)

    e = episode_idx
    x = scaler.transform(teb.flat_features()).reshape(teb.n_episodes, teb.horizon, -1)[e]
    w = proto.weights(x)                            # [L, K]
    raw = proto.prototypes * scaler.std + scaler.mean
    wsurf = (w.mean(0) @ raw[:, :4])               # [4] episode-mean weighted surface params

    moneyness = np.linspace(0.8, 1.2, 40)
    tenor = np.array([7, 30, 60, 90, 180, 365])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title, fontsize=13, y=1.01)

    # (a) Surface heatmap
    Z = _surface(wsurf, moneyness, tenor)
    im = axes[0, 0].imshow(Z, aspect="auto", origin="lower", cmap="viridis",
                           extent=[0.8, 1.2, 0, len(tenor) - 1])
    axes[0, 0].set_yticks(range(len(tenor)))
    axes[0, 0].set_yticklabels(tenor)
    axes[0, 0].set_title("(a) Episode surface (prototype-weighted blend)")
    axes[0, 0].set_xlabel("moneyness K/F")
    axes[0, 0].set_ylabel("tenor (days)")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8, label="IV")

    # (b) Prototype activation heatmap — top 3 labelled
    axes[0, 1].imshow(w.T, aspect="auto", origin="lower", cmap="magma",
                      vmin=0, vmax=w.max())
    avg_w = w.mean(0)
    top3 = np.argsort(avg_w)[::-1][:3]
    axes[0, 1].set_title("(b) Prototype activations across episode")
    axes[0, 1].set_xlabel("rebalance step")
    axes[0, 1].set_ylabel("prototype k")
    axes[0, 1].set_yticks(range(proto.prototypes.shape[0]))
    top_str = "  ".join(f"P{k}:{avg_w[k]:.2f}" for k in top3)
    axes[0, 1].annotate(f"top: {top_str}", xy=(0.02, 1.01), xycoords="axes fraction",
                        fontsize=8, ha="left")

    # (c) Holdings decomposition
    axes[1, 0].plot(base_h[e][:, 0], label="Δ-vega base (shares)", color="#55a868", lw=1.5)
    axes[1, 0].plot(out["holdings"][e][:, 0], label="prototype (shares)", color="#8172b3", lw=1.5)
    resid = out["holdings"][e][:, 0] - base_h[e][:, 0]
    axes[1, 0].fill_between(range(teb.horizon), base_h[e][:, 0], out["holdings"][e][:, 0],
                            color="#c44e52", alpha=0.3, label="residual")
    axes[1, 0].axhline(0, color="k", lw=0.6)
    axes[1, 0].set_title("(c) Share holdings: prototype = Δ-vega + learned residual")
    axes[1, 0].set_xlabel("rebalance step")
    axes[1, 0].set_ylabel("underlying shares")
    axes[1, 0].legend(fontsize=8)

    # (d) Cumulative P&L
    cum_p = np.cumsum(-np.diff(teb.v_liab[e]) * teb.config.notional
                      + out["holdings"][e][:, 0] * np.diff(teb.spot[e])
                      + out["holdings"][e][:, 1] * np.diff(teb.o_hedge[e]))
    cum_d = np.cumsum(-np.diff(teb.v_liab[e]) * teb.config.notional
                      + base_h[e][:, 0] * np.diff(teb.spot[e])
                      + base_h[e][:, 1] * np.diff(teb.o_hedge[e]))
    axes[1, 1].plot(cum_d, label="delta-vega", color="#55a868", lw=1.5)
    axes[1, 1].plot(cum_p, label="prototype", color="#8172b3", lw=1.5)
    axes[1, 1].axhline(0, color="k", lw=0.6, ls="--")
    axes[1, 1].set_title("(d) Cumulative hedged P&L on this episode")
    axes[1, 1].set_xlabel("rebalance step")
    axes[1, 1].set_ylabel("cumulative P&L")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="spy")
    ap.add_argument("--stress-year", type=int, default=2020)
    ap.add_argument("--calm-year", type=int, default=2019)
    ap.add_argument("--reports-dir", default="reports_real")
    args = ap.parse_args()

    figs = ROOT / args.reports_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.universe}...")
    obj = _load(args.universe)
    bank, years_all = obj["bank"], obj.get("years")

    proto, scaler, teb, sc_te, sp = _fit_winner(bank)
    years_test = years_all[sp.test] if years_all is not None else np.zeros(teb.n_episodes, dtype=int)

    for mode, year in [("stress", args.stress_year), ("calm", args.calm_year)]:
        print(f"picking {mode} episode (year={year})...")
        idx = _pick_episode(teb, years_test, year, mode)
        regime = "stress" if teb.regime_start[idx] == 1 else "calm"
        title = (f"{args.universe.upper()} {year} {mode} episode (idx={idx}, "
                 f"bank_regime={regime})")
        fname = figs / f"trade_anatomy_{args.universe}_{mode}_{year}.png"
        _anatomy_figure(proto, scaler, teb, sc_te, idx, title, fname)

    print("done.")


if __name__ == "__main__":
    main()
