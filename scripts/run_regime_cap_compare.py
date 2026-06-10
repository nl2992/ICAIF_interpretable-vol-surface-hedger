"""Regime-conditional cap: vol-acceleration vs vol-level walk-forward comparison.

Reviewer point: the COVID-2020 fold is 4x worse than delta-vega even after the
level cap (17.96 vs 4.12). The cap only activates once vol is already high, but
COVID's worst damage happened during the vol SPIKE onset. An acceleration-based
cap (tighten when d(rv)/dt is large) activates earlier.

This replicates the exact walk-forward fold structure (n_train, n_test from the
existing results file) and trains the prototype under the acceleration cap, then
reports year-by-year CVaR95 for both cap types side-by-side.

Output: reports_real/tables/regime_cap_compare_spy.csv
        reports_real/figures/regime_cap_compare_spy.png
"""
from __future__ import annotations

import pathlib as _pl
import pickle
import sys

_ROOT = _pl.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ivsh.training.train import (
    TrainConfig, fit_prototype, make_standardizer,
    realized_vol_scale, realized_vol_accel_scale,
)
from ivsh.training.objective import cvar_from_pnl
from ivsh.utils.splits import subset
from ivsh.baselines.policies import delta_vega_hedge

UNIVERSE = "spy"
ALPHA = 0.95
ANCHOR = True
MAX_ITER = 250   # match original walkforward run_real_analysis.py default
SEED = 7
RSCALE = 1.5     # match original walkforward run_real_analysis.py default
N_PROTO = 8


def _train_cfg(seed=SEED):
    return TrainConfig(cvar_alpha=ALPHA, cvar_weight=1.0, max_iter=MAX_ITER,
                       n_prototypes=N_PROTO, action_scale=RSCALE, anchor=ANCHOR, seed=seed)


def run_fold(bank, train_idx, test_idx, cap_mode: str):
    """Run one walk-forward fold under cap_mode='level'|'accel'|'none'."""
    order = train_idx[np.argsort(bank.start_days[train_idx])]
    cut = int(len(order) * 0.85)
    trb = subset(bank, order[:cut])
    vlb = subset(bank, order[cut:])
    teb = subset(bank, test_idx)
    scaler = make_standardizer(trb)
    vol_floor = 0.25

    if cap_mode == "level":
        sc_tr, ref = realized_vol_scale(trb, floor=vol_floor)
        sc_vl, _ = realized_vol_scale(vlb, ref=ref, floor=vol_floor) if vlb.n_episodes else (None, ref)
        sc_te, _ = realized_vol_scale(teb, ref=ref, floor=vol_floor)
    elif cap_mode == "accel":
        sc_tr, ref = realized_vol_accel_scale(trb, floor=vol_floor, accel_gamma=3.0)
        sc_vl, _ = realized_vol_accel_scale(vlb, ref=ref, floor=vol_floor) if vlb.n_episodes else (None, ref)
        sc_te, _ = realized_vol_accel_scale(teb, ref=ref, floor=vol_floor)
    else:
        sc_tr = sc_vl = sc_te = None

    policy, _, _ = fit_prototype(trb, scaler, _train_cfg(SEED),
                                 val_bank=vlb, residual_scale=sc_tr, val_residual_scale=sc_vl)

    # Evaluate using the true P&L (anchored = +delta-vega base)
    E, L = teb.n_episodes, teb.horizon
    x_te = scaler.transform(teb.flat_features())
    resid = policy.predict_holdings(x_te).reshape(E, L, -1)
    if sc_te is not None:
        resid = resid * sc_te[:, :, None]
    if ANCHOR:
        base = delta_vega_hedge(teb)
        holdings = resid + base
    else:
        holdings = resid
    pnl = teb.episode_pnl(holdings)
    return float(cvar_from_pnl(pnl, ALPHA)), float(pnl.mean())


def build_folds(bank, wf_csv_path):
    """Reconstruct year-fold indices from the existing walkforward CSV.

    Expanding-window: train_idx = all episodes before the test block (growing);
    test_idx = the n_test-episode block for that year. The initial pre-test training
    size equals n_train of the first row.
    """
    wf = pd.read_csv(wf_csv_path)
    order = np.argsort(bank.start_days)  # chronological episode order
    pos = int(wf.iloc[0]["n_train"])     # skip initial training-only episodes
    folds = []
    for _, row in wf.iterrows():
        yr = int(row["test_year"])
        n_te = int(row["n_test"])
        train_idx = order[:pos]          # expanding: all prior episodes
        test_idx = order[pos: pos + n_te]
        folds.append((yr, train_idx, test_idx))
        pos += n_te                      # advance test cursor
    return folds


def main():
    with open(_ROOT / "artifacts" / f"bank_{UNIVERSE}.pkl", "rb") as f:
        bank = pickle.load(f)["bank"]

    wf_csv = _ROOT / "reports_real" / "tables" / "walkforward_cvar.csv"
    folds = build_folds(bank, wf_csv)
    print(f"Loaded {len(folds)} walk-forward folds")

    rows = []
    for mode in ("level", "accel"):
        for yr, train_idx, test_idx in folds:
            print(f"  year={yr} cap={mode} n_tr={len(train_idx)} n_te={len(test_idx)} ...", flush=True)
            cvar95, mpnl = run_fold(bank, train_idx, test_idx, mode)
            rows.append({"test_year": yr, "cap_mode": mode,
                         "cvar95": round(cvar95, 4), "mean_pnl": round(mpnl, 4)})
            print(f"    cvar95={cvar95:.3f}", flush=True)

    df = pd.DataFrame(rows)
    out_dir = _ROOT / "reports_real"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "tables" / f"regime_cap_compare_{UNIVERSE}.csv", index=False)

    pivot = df.pivot(index="test_year", columns="cap_mode", values="cvar95")
    if "level" in pivot and "accel" in pivot:
        pivot["delta_cap"] = pivot["level"] - pivot["accel"]
    print("\n=== CVaR95 by year, lower is better ===")
    print(pivot.round(3).to_string())

    # Figure
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    yrs = list(pivot.index)
    x = np.arange(len(yrs))
    w = 0.35
    if "level" in pivot.columns:
        ax.bar(x - w/2, pivot["level"].values, w, label="Level cap (existing)", color="#002B7F", alpha=0.85)
    if "accel" in pivot.columns:
        ax.bar(x + w/2, pivot["accel"].values, w, label="Accel cap (new)", color="#C8102E", alpha=0.85)
    if 2020 in yrs:
        ci = yrs.index(2020)
        ax.axvspan(ci - 0.5, ci + 0.5, alpha=0.10, color="orange", label="COVID-2020")
    ax.set_xticks(x); ax.set_xticklabels(yrs, fontsize=9)
    ax.set_ylabel("CVaR95 tail loss (lower = better)", fontsize=9)
    ax.set_title("Regime-conditional cap: level vs acceleration\n(prototype, SPY, walk-forward)", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"regime_cap_compare_{UNIVERSE}.pdf", dpi=200)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
