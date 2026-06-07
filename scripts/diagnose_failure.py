"""Diagnose *why* the initial anchored-residual prototype failed on QQQ.

Produces ``reports_real/why_initial_failed.md`` plus figures:
  (1) Residual decomposition — magnitude/turnover and tail-loss contribution of the
      uncapped residual, by regime, on each universe (shows the residual ADDS tail
      risk in stress on QQQ).
  (2) Validation->test distribution shift (val is calm, test spans COVID/2022).
  (3) Per-knob ablation from the grid log (which lever moved QQQ loss -> tie).

    python scripts/diagnose_failure.py --universe spy qqq
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
from ivsh.evaluation.backtest import run_baseline, run_policy
from ivsh.training.objective import cvar_from_pnl
from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer
from ivsh.utils.splits import chronological_split, subset

ROOT = _pl.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def _load(uni):
    with open(ARTIFACTS / f"bank_{uni}.pkl", "rb") as f:
        return pickle.load(f)["bank"]


def _tail_mask(pnl, alpha=0.95):
    loss = -np.asarray(pnl)
    return loss >= np.quantile(loss, alpha)


def diagnose(uni, lines, figs):
    bank = _load(uni)
    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    scaler = make_standardizer(trb)
    # the original (uncapped, unregularised) anchored prototype
    tc = TrainConfig(n_prototypes=8, l2=1e-3, max_iter=250, anchor=True, action_scale=1.5, seed=7)
    p, _, _ = fit_prototype(trb, scaler, tc, val_bank=vlb)
    out = run_policy(p, teb, scaler, anchor=True)
    base = delta_vega_hedge(teb)
    resid = out["holdings"] - base  # [E,L,2]
    pnl_proto = out["pnl"]
    pnl_dv = run_baseline(teb, "delta_vega")["pnl"]

    tail = _tail_mask(pnl_proto)
    resid_mag = np.abs(resid).sum(axis=2).mean(axis=1)  # per-episode residual size
    lines.append(f"\n### {uni.upper()}\n")
    lines.append(f"- test CVaR95: prototype **{cvar_from_pnl(pnl_proto):.3f}** vs "
                 f"delta-vega **{cvar_from_pnl(pnl_dv):.3f}** "
                 f"(excess {cvar_from_pnl(pnl_proto)-cvar_from_pnl(pnl_dv):+.3f}).")
    lines.append(f"- mean |residual| over all episodes: {resid_mag.mean():.3f}; "
                 f"over the CVaR95 tail episodes: **{resid_mag[tail].mean():.3f}** "
                 f"({resid_mag[tail].mean()/max(resid_mag.mean(),1e-9):.1f}x larger).")
    # how often does the residual make the tail WORSE than the delta-vega base?
    worse = (pnl_proto[tail] < pnl_dv[tail]).mean()
    lines.append(f"- in the tail, the residual leaves P&L *worse than the delta-vega base* "
                 f"in **{100*worse:.0f}%** of tail episodes.")

    # figure: residual magnitude, tail vs non-tail
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([resid_mag[~tail], resid_mag[tail]], labels=["non-tail", "CVaR95 tail"])
    ax.set_ylabel("mean |residual| per episode")
    ax.set_title(f"{uni.upper()}: residual is larger in the tail")
    fig.tight_layout(); fig.savefig(figs / f"diag_residual_{uni}.png", dpi=130); plt.close(fig)

    # val->test shift on realised vol
    idx = bank.feature_names.index("realized_vol")
    rv_val = vlb.features[:, :, idx].mean()
    rv_test = teb.features[:, :, idx].mean()
    lines.append(f"- mean realised vol: validation {rv_val:.3f} vs test **{rv_test:.3f}** "
                 f"({rv_test/max(rv_val,1e-9):.2f}x) — the model is selected on a calmer "
                 f"regime than it is tested on.")
    return rv_val, rv_test


def ablation(lines, figs):
    grid = ROOT / "reports_real" / "grid" / "grid_results.csv"
    if not grid.exists():
        lines.append("\n_(grid_results.csv not found — run scripts/grid_search.py for the "
                     "per-knob ablation.)_\n")
        return
    df = pd.read_csv(grid)
    lines.append("\n## What fixed it (per-knob, validation excess over delta-vega)\n")
    for knob in ["residual_l2", "vol_floor", "action_scale", "cvar_weight"]:
        if knob not in df:
            continue
        t = df.groupby(knob)["val_excess"].mean().round(4)
        lines.append(f"- **{knob}**: " + ", ".join(f"{k}->{v:+.3f}" for k, v in t.items()))
    # figure: best val_excess per hypothesis
    best = df.groupby("hypo")["val_excess"].min().sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(best.index, best.values, color="#4c72b0", edgecolor="k")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("best mean validation excess-CVaR over delta-vega (<=0 is a tie/win)")
    ax.set_title("Which hypothesis closes the gap?")
    fig.tight_layout(); fig.savefig(figs / "diag_ablation.png", dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", nargs="+", default=["spy", "qqq"])
    args = ap.parse_args()
    figs = ROOT / "reports_real" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    lines = ["# Why the initial anchored-residual prototype failed (and what fixes it)\n",
             "The v1 prototype anchored a learned residual on the delta-vega hedge. On QQQ the "
             "residual *added* tail risk. This note quantifies the mechanism.\n",
             "## Residual decomposition and validation->test shift"]
    for u in args.universe:
        diagnose(u, lines, figs)
    ablation(lines, figs)
    out = ROOT / "reports_real" / "why_initial_failed.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
