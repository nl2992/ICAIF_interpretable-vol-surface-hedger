"""ProtoHedge-style scalar-Greeks prototype baseline on the synthetic market.

This is an apples-to-apples scalar-state prototype comparator: same action space,
same environment and objective, but the prototype medoids see only Greek/book
features rather than the volatility-surface state.

Example:
    python scripts/run_protohedge_baseline.py --config configs/experiment.yaml --seeds 5
"""

from __future__ import annotations

import argparse
import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from ivsh.evaluation.backtest import GREEK_FEATURES, run_policy
from ivsh.evaluation.metrics import compute_metrics
from ivsh.pipeline import build_data, load_config
from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer
from ivsh.utils.splits import select_features

ROOT = _pl.Path(__file__).resolve().parents[1]


def _fit_eval(trb, vlb, teb, *, seed: int, max_iter: int):
    scaler = make_standardizer(trb)
    cfg = TrainConfig(n_prototypes=8, l2=1e-3, max_iter=max_iter, seed=seed)
    proto, _, _ = fit_prototype(trb, scaler, cfg, val_bank=vlb)
    res = run_policy(proto, teb, scaler)
    return compute_metrics(res["pnl"], res["turnover"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--reports-dir", default="reports")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    data = build_data(cfg)
    max_iter = args.max_iter or cfg.proto_train.max_iter
    rows = []
    for i in range(args.seeds):
        seed = cfg.seed + i
        full = _fit_eval(data["train"], data["val"], data["test"], seed=seed, max_iter=max_iter)
        tr_g = select_features(data["train"], GREEK_FEATURES)
        vl_g = select_features(data["val"], GREEK_FEATURES)
        te_g = select_features(data["test"], GREEK_FEATURES)
        greek = _fit_eval(tr_g, vl_g, te_g, seed=seed, max_iter=max_iter)
        rows.append({"seed": seed, "model": "surface_proto", **full})
        rows.append({"seed": seed, "model": "protohedge_scalar_greeks", **greek})
        print(f"seed {seed}: surface CVaR95={full['cvar_95']:.3f}; scalar ProtoHedge CVaR95={greek['cvar_95']:.3f}")

    out = ROOT / args.reports_dir / "tables" / "protohedge_baseline.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
