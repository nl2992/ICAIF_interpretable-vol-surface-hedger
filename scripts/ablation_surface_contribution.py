"""Surface-vs-Greeks contribution study on cached real-data banks.

Example:
    python scripts/ablation_surface_contribution.py --universes spy qqq slv
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

import numpy as np
import pandas as pd

from ivsh.evaluation.backtest import GREEK_FEATURES, SURFACE_FEATURES, run_policy
from ivsh.evaluation.metrics import compute_metrics
from ivsh.evaluation.stats import paired_bootstrap_diff
from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer
from ivsh.utils.splits import chronological_split, select_features, subset

ROOT = _pl.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

FEATURE_SETS = {
    "greeks_only": GREEK_FEATURES,
    "surface_only": SURFACE_FEATURES,
    "full": None,
}


def _load_bank(universe: str):
    with open(ARTIFACTS / f"bank_{universe}.pkl", "rb") as f:
        return pickle.load(f)["bank"]


def _prepare(bank, feature_set: str):
    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    names = FEATURE_SETS[feature_set]
    if names is not None:
        trb, vlb, teb = select_features(trb, names), select_features(vlb, names), select_features(teb, names)
    return trb, vlb, teb


def run_one(universe: str, feature_set: str, seed: int, max_iter: int):
    trb, vlb, teb = _prepare(_load_bank(universe), feature_set)
    scaler = make_standardizer(trb)
    cfg = TrainConfig(
        n_prototypes=8,
        l2=1e-3,
        max_iter=max_iter,
        anchor=True,
        action_scale=1.5,
        seed=seed,
    )
    proto, _, _ = fit_prototype(trb, scaler, cfg, val_bank=vlb)
    res = run_policy(proto, teb, scaler, anchor=True)
    row = {"universe": universe, "feature_set": feature_set, "seed": seed}
    row.update(compute_metrics(res["pnl"], res["turnover"]))
    return row, res["pnl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universes", nargs="+", default=["spy", "qqq"])
    ap.add_argument("--feature-set", nargs="+", default=["greeks_only", "surface_only", "full"],
                    choices=sorted(FEATURE_SETS))
    ap.add_argument("--seed", type=int, nargs="+", default=[7],
                    help="one or more seeds; results are averaged and CIs are bootstrapped")
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--reports-dir", default="reports_real")
    args = ap.parse_args()
    seeds = args.seed

    outdir = ROOT / args.reports_dir / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    # pnls[u, fs] = list of per-seed pnl arrays (to be concatenated for bootstrap)
    pnl_by_seed: dict[tuple[str, str], list[np.ndarray]] = {}
    for u in args.universes:
        if not (ARTIFACTS / f"bank_{u}.pkl").exists():
            print(f"[skip] missing cached bank: artifacts/bank_{u}.pkl")
            continue
        for fs in args.feature_set:
            pnl_list = []
            for seed in seeds:
                row, pnl = run_one(u, fs, seed, args.max_iter)
                rows.append(row)
                pnl_list.append(pnl)
                print(f"[{u} {fs} seed={seed}] CVaR95={row['cvar_95']:.3f} utility={row['utility']:.3f}")
            pnl_by_seed[(u, fs)] = pnl_list

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "surface_contribution.csv", index=False)

    # Per-seed summary — use the same cvar_from_pnl as compute_metrics
    from ivsh.training.objective import cvar_from_pnl as _cvar
    agg_rows = []
    for (u, fs), pnl_list in pnl_by_seed.items():
        cvars = [float(_cvar(pnl)) for pnl in pnl_list]
        agg_rows.append({"universe": u, "feature_set": fs,
                         "n_seeds": len(seeds),
                         "cvar95_mean": float(np.mean(cvars)),
                         "cvar95_std": float(np.std(cvars))})
    pd.DataFrame(agg_rows).to_csv(outdir / "surface_contribution_multiseed.csv", index=False)

    # Bootstrap CIs using all seeds pooled
    gaps = []
    for u in sorted({u for u, _ in pnl_by_seed}):
        if (u, "full") in pnl_by_seed and (u, "greeks_only") in pnl_by_seed:
            pnl_full = np.concatenate(pnl_by_seed[(u, "full")])
            pnl_greeks = np.concatenate(pnl_by_seed[(u, "greeks_only")])
            bs = paired_bootstrap_diff(pnl_full, pnl_greeks, stat="cvar")
            gaps.append({
                "universe": u,
                "comparison": "full_minus_greeks_only",
                "n_seeds": len(seeds),
                "dcvar95": bs["diff"],
                "ci_low": bs["ci_low"],
                "ci_high": bs["ci_high"],
                "p_bootstrap": bs["p_two_sided"],
            })
    pd.DataFrame(gaps).to_csv(outdir / "surface_marginal_contribution.csv", index=False)
    print(f"wrote {outdir / 'surface_contribution.csv'} + multiseed summary + marginal contribution")


if __name__ == "__main__":
    main()
