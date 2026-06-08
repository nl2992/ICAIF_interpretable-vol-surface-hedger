"""Catalogue prototype regimes from a cached bank and a freshly fitted winner."""

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

from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer
from ivsh.utils.splits import chronological_split, subset

ROOT = _pl.Path(__file__).resolve().parents[1]


def _label(row) -> str:
    bits = []
    if row["iv_level"] >= 0.28:
        bits.append("high-vol shock")
    elif row["iv_level"] <= 0.15:
        bits.append("calm/low-vol")
    else:
        bits.append("mid-vol")
    if row["term_slope"] < -0.01:
        bits.append("front-end inversion")
    elif row["term_slope"] > 0.005:
        bits.append("upward term structure")
    if row["skew"] < -0.46:
        bits.append("left-tail skew steepening")
    if row["curvature"] > 0.9:
        bits.append("high smile curvature")
    return ", ".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="spy")
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--reports-dir", default="reports_real")
    args = ap.parse_args()

    with open(ROOT / "artifacts" / f"bank_{args.universe}.pkl", "rb") as f:
        obj = pickle.load(f)
    bank = obj["bank"]
    years = obj.get("years")
    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    scaler = make_standardizer(trb)
    cfg = TrainConfig(n_prototypes=8, l2=1e-3, max_iter=args.max_iter, anchor=True, action_scale=1.5, seed=7)
    proto, _, _ = fit_prototype(trb, scaler, cfg, val_bank=vlb)

    x = scaler.transform(teb.flat_features()).reshape(teb.n_episodes, teb.horizon, -1)
    weights = proto.weights(x.reshape(teb.n_episodes * teb.horizon, -1)).reshape(teb.n_episodes, teb.horizon, -1)
    winner = weights.mean(axis=1).argmax(axis=1)
    raw = proto.prototypes * scaler.std + scaler.mean
    rows = []
    for k in range(proto.prototypes.shape[0]):
        mask = winner == k
        row = {
            "universe": args.universe,
            "prototype": f"P{k}",
            "top_episode_share": float(mask.mean()) if len(mask) else 0.0,
            "stress_episode_share": float(teb.regime_start[mask].mean()) if mask.any() else 0.0,
            "mean_activation": float(weights[:, :, k].mean()),
            "iv_level": float(raw[k, 0]),
            "skew": float(raw[k, 1]),
            "curvature": float(raw[k, 2]),
            "term_slope": float(raw[k, 3]),
        }
        if years is not None and mask.any():
            test_years = years[sp.test]
            vc = pd.Series(test_years[mask]).value_counts()
            row["top_year"] = int(vc.index[0])
            row["top_year_count"] = int(vc.iloc[0])
        row["regime_label"] = _label(row)
        rows.append(row)

    out = ROOT / args.reports_dir / "tables" / f"prototype_regime_audit_{args.universe}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
