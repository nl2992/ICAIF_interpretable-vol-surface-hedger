"""Multi-hypothesis grid search for a robust, interpretable hedger.

Loads the cached SPY/QQQ episode banks (``scripts/cache_banks.py``), sweeps the
H1--H7 hypothesis grid in parallel across CPU cores, and selects a winner **on the
validation split only** (minimising mean validation excess-CVaR over delta--vega
across both universes), then confirms it once on the test split of both universes.

Anti-overfitting protocol: test is never used for selection; the full grid is
logged to ``reports_real/grid/grid_results.csv`` for transparency.

    python scripts/grid_search.py --universe spy qqq --procs 20
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
import os
import pickle
import warnings
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ivsh.baselines.policies import delta_vega_hedge
from ivsh.evaluation.backtest import run_baseline, run_policy, run_policy_ensemble
from ivsh.training.objective import cvar_from_pnl
from ivsh.training.train import (
    TrainConfig, fit_prototype, make_standardizer, realized_vol_scale,
)
from ivsh.utils.splits import chronological_split, select_features, stress_resample_index, subset

ROOT = _pl.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

SURFACE_ONLY = ("surf_level", "surf_skew", "surf_curv", "surf_slope", "atm_iv_short",
                "atm_iv_long", "term_slope", "realized_vol", "ret_5d", "dlevel_1d")
GREEKS_ONLY = ("liab_delta", "liab_vega", "liab_gamma", "liab_logmoney", "liab_ttm",
               "hedge_delta", "hedge_vega")
SEEDS_SEL = [7, 13, 23]
SEEDS_ENS = [7, 13, 23, 42, 2025]

_BANKS: dict = {}  # populated per worker by _init_worker


# --------------------------------------------------------------------------- #
# Grid definition
# --------------------------------------------------------------------------- #
def _cfg(name, hypo, *, action_scale=1.5, l2=1e-3, n_prototypes=8, cvar_weight=1.0,
         cvar_alpha=0.95, residual_l2=0.0, vol_floor=None, features=None,
         stress_power=0.0, ensemble=False):
    return dict(name=name, hypo=hypo, action_scale=action_scale, l2=l2,
                n_prototypes=n_prototypes, cvar_weight=cvar_weight, cvar_alpha=cvar_alpha,
                residual_l2=residual_l2, vol_floor=vol_floor, features=features,
                stress_power=stress_power, ensemble=ensemble)


def build_grid():
    g = [_cfg("baseline", "baseline")]
    # H1 residual regularisation: action_scale x vol_floor, plus an l2 sweep
    for a, vf in product([0.5, 1.0, 1.5], [None, 0.5, 0.25, 0.1]):
        g.append(_cfg(f"H1_as{a}_vf{vf}", "H1", action_scale=a, vol_floor=vf))
    for l2 in [1e-2, 3e-2, 1e-1]:
        g.append(_cfg(f"H1_l2{l2}", "H1", l2=l2))
    # H2 tail-weighted objective
    for cw, al in product([1.0, 3.0, 10.0], [0.95, 0.975, 0.99]):
        if (cw, al) != (1.0, 0.95):
            g.append(_cfg(f"H2_cw{cw}_a{al}", "H2", cvar_weight=cw, cvar_alpha=al))
    # H3 prototype count
    for k in [6, 12, 16]:
        g.append(_cfg(f"H3_K{k}", "H3", n_prototypes=k))
    # H4 do-no-harm shrink-to-base
    for rl2 in [1.0, 10.0, 100.0, 1000.0]:
        g.append(_cfg(f"H4_rl2{rl2}", "H4", residual_l2=rl2))
    # H5 feature transfer
    g.append(_cfg("H5_surface_only", "H5", features=SURFACE_ONLY))
    g.append(_cfg("H5_greeks_only", "H5", features=GREEKS_ONLY))
    # combo of the most promising levers
    for a, vf, rl2, cw in product([0.5, 1.0], [0.25, None], [0.0, 10.0, 100.0], [1.0, 3.0]):
        g.append(_cfg(f"C_as{a}_vf{vf}_rl2{rl2}_cw{cw}", "combo", action_scale=a,
                      vol_floor=vf, residual_l2=rl2, cvar_weight=cw))
    # H6 seed ensemble (applied to a regularised base config)
    g.append(_cfg("H6_ensemble", "H6", action_scale=1.0, vol_floor=0.25, ensemble=True))
    # H7 stress reweighting
    for p in [1.0, 2.0, 3.0]:
        g.append(_cfg(f"H7_stress{p}", "H7", stress_power=p, vol_floor=0.25, action_scale=1.0))
    # de-dup by name
    seen, out = set(), []
    for c in g:
        if c["name"] not in seen:
            seen.add(c["name"]); out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Fit / evaluate one (config, universe)
# --------------------------------------------------------------------------- #
def _init_worker(universe_paths):
    for name, path in universe_paths.items():
        with open(path, "rb") as f:
            _BANKS[name] = pickle.load(f)["bank"]


def _prep_bank(bank, features):
    return select_features(bank, features) if features else bank


def _scales(trb, vlb, teb, vol_floor):
    if vol_floor is None or "realized_vol" not in trb.feature_names:
        return None, None, None
    sc_tr, ref = realized_vol_scale(trb, floor=vol_floor)
    sc_vl = realized_vol_scale(vlb, ref=ref, floor=vol_floor)[0] if vlb.n_episodes else None
    sc_te = realized_vol_scale(teb, ref=ref, floor=vol_floor)[0]
    return sc_tr, sc_vl, sc_te


def eval_config(task):
    cfg, uni = task
    bank = _prep_bank(_BANKS[uni], cfg["features"])
    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    if cfg["stress_power"] > 0:
        trb = subset(trb, stress_resample_index(trb, cfg["stress_power"]))

    scaler = make_standardizer(trb)
    sc_tr, sc_vl, sc_te = _scales(trb, vlb, teb, cfg["vol_floor"])
    tc = TrainConfig(n_prototypes=cfg["n_prototypes"], l2=cfg["l2"], max_iter=250,
                     anchor=True, action_scale=cfg["action_scale"],
                     cvar_weight=cfg["cvar_weight"], cvar_alpha=cfg["cvar_alpha"],
                     residual_l2=cfg["residual_l2"])
    seeds = SEEDS_ENS if cfg["ensemble"] else SEEDS_SEL

    policies = []
    val_cvars, test_cvars = [], []
    for s in seeds:
        tc.seed = s
        p, _, _ = fit_prototype(trb, scaler, tc, val_bank=vlb,
                                residual_scale=sc_tr, val_residual_scale=sc_vl)
        policies.append(p)
        if not cfg["ensemble"]:
            val_cvars.append(cvar_from_pnl(run_policy(p, vlb, scaler, anchor=True,
                                                      residual_scale=sc_vl)["pnl"]))
            test_cvars.append(cvar_from_pnl(run_policy(p, teb, scaler, anchor=True,
                                                       residual_scale=sc_te)["pnl"]))
    if cfg["ensemble"]:
        val_cvar = cvar_from_pnl(run_policy_ensemble(policies, vlb, scaler, anchor=True,
                                                     residual_scale=sc_vl)["pnl"])
        test_cvar = cvar_from_pnl(run_policy_ensemble(policies, teb, scaler, anchor=True,
                                                      residual_scale=sc_te)["pnl"])
        val_std = test_std = 0.0
    else:
        val_cvar, test_cvar = float(np.mean(val_cvars)), float(np.mean(test_cvars))
        val_std, test_std = float(np.std(val_cvars)), float(np.std(test_cvars))

    dv_val = cvar_from_pnl(run_baseline(vlb, "delta_vega")["pnl"])
    dv_test = cvar_from_pnl(run_baseline(teb, "delta_vega")["pnl"])
    return {
        "config": cfg["name"], "hypo": cfg["hypo"], "universe": uni,
        "action_scale": cfg["action_scale"], "l2": cfg["l2"], "K": cfg["n_prototypes"],
        "cvar_weight": cfg["cvar_weight"], "cvar_alpha": cfg["cvar_alpha"],
        "residual_l2": cfg["residual_l2"], "vol_floor": cfg["vol_floor"],
        "features": "full" if not cfg["features"] else ("surface" if cfg["features"] == SURFACE_ONLY else "greeks"),
        "stress_power": cfg["stress_power"], "ensemble": cfg["ensemble"],
        "val_cvar": val_cvar, "val_dv": dv_val, "val_excess": val_cvar - dv_val,
        "test_cvar": test_cvar, "test_dv": dv_test, "test_excess": test_cvar - dv_test,
        "test_seed_std": test_std,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", nargs="+", default=["spy", "qqq"])
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default="reports_real/grid")
    args = ap.parse_args()

    upaths = {u: ARTIFACTS / f"bank_{u}.pkl" for u in args.universe}
    for u, p in upaths.items():
        if not p.exists():
            ap.error(f"missing cached bank {p}; run scripts/cache_banks.py first")

    grid = build_grid()
    tasks = [(c, u) for c in grid for u in args.universe]
    print(f"grid: {len(grid)} configs x {len(args.universe)} universes = {len(tasks)} fits "
          f"on {args.procs} procs")

    import multiprocessing as mp
    with mp.Pool(args.procs, initializer=_init_worker, initargs=(upaths,)) as pool:
        rows = pool.map(eval_config, tasks)
    df = pd.DataFrame(rows)

    out = _pl.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "grid_results.csv", index=False)

    # Selection: minimise mean validation excess-over-delta-vega across universes.
    sel = (df.groupby("config")
             .agg(mean_val_excess=("val_excess", "mean"),
                  max_val_excess=("val_excess", "max"),
                  hypo=("hypo", "first"))
             .sort_values("mean_val_excess"))
    sel.to_csv(out / "grid_selection.csv")
    winner = sel.index[0]
    print("\n=== top 8 configs by mean validation excess-over-delta-vega ===")
    print(sel.head(8).round(4).to_string())

    # Confirm winner once on test (both universes).
    w = df[df["config"] == winner].set_index("universe")
    print(f"\n=== WINNER (val-selected): {winner} ===")
    bar_met = True
    for u in args.universe:
        r = w.loc[u]
        ok = r["test_excess"] <= 0.05 * max(r["test_dv"], 1e-9) + 1e-9  # tie-or-beat (5% tol)
        bar_met &= bool(ok)
        print(f"  [{u}] test prototype={r['test_cvar']:.3f} vs delta_vega={r['test_dv']:.3f}"
              f"  excess={r['test_excess']:+.3f}  {'OK' if ok else 'MISS'}")
    print(f"\nBar (tie-or-beat delta-vega on ALL universes): {'MET' if bar_met else 'NOT MET'}")
    w.to_csv(out / "grid_winner.csv")


if __name__ == "__main__":
    main()
