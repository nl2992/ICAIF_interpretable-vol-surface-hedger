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
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--reports-dir", default="reports_real")
    args = ap.parse_args()

    outdir = ROOT / args.reports_dir / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    pnls: dict[tuple[str, str], np.ndarray] = {}
    for u in args.universes:
        if not (ARTIFACTS / f"bank_{u}.pkl").exists():
            print(f"[skip] missing cached bank: artifacts/bank_{u}.pkl")
            continue
        for fs in args.feature_set:
            row, pnl = run_one(u, fs, args.seed, args.max_iter)
            rows.append(row)
            pnls[(u, fs)] = pnl
            print(f"[{u} {fs}] CVaR95={row['cvar_95']:.3f} utility={row['utility']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "surface_contribution.csv", index=False)

    gaps = []
    for u in sorted({u for u, _ in pnls}):
        if (u, "full") in pnls and (u, "greeks_only") in pnls:
            bs = paired_bootstrap_diff(pnls[(u, "full")], pnls[(u, "greeks_only")], stat="cvar")
            gaps.append({
                "universe": u,
                "comparison": "full_minus_greeks_only",
                "dcvar95": bs["diff"],
                "ci_low": bs["ci_low"],
                "ci_high": bs["ci_high"],
                "p_bootstrap": bs["p_two_sided"],
            })
    pd.DataFrame(gaps).to_csv(outdir / "surface_marginal_contribution.csv", index=False)
    print(f"wrote {outdir / 'surface_contribution.csv'}")


if __name__ == "__main__":
    main()
