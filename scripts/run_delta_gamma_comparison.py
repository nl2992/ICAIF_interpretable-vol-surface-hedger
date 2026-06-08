"""Run delta-gamma-vega baseline on all cached banks and compare to existing results.

Loads each artifacts/bank_<universe>.pkl, evaluates unhedged / delta / delta-vega /
delta-gamma-vega, and writes a comparison CSV.

Example:
    python scripts/run_delta_gamma_comparison.py --universes spy qqq slv spx
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

import pandas as pd

from ivsh.evaluation.backtest import run_baseline
from ivsh.evaluation.metrics import compute_metrics
from ivsh.utils.splits import chronological_split, subset

ROOT = _pl.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

METHODS = ["unhedged", "delta", "delta_vega", "delta_gamma_vega"]


def run_universe(universe: str) -> list[dict]:
    path = ARTIFACTS / f"bank_{universe}.pkl"
    if not path.exists():
        print(f"[skip] missing {path}")
        return []
    with open(path, "rb") as f:
        obj = pickle.load(f)
    bank = obj["bank"]
    sp = chronological_split(bank)
    teb = subset(bank, sp.test)
    rows = []
    for method in METHODS:
        try:
            res = run_baseline(teb, method)
            row = {"universe": universe, "method": method}
            row.update(compute_metrics(res["pnl"], res["turnover"]))
            rows.append(row)
            print(f"  [{universe}] {method}: CVaR95={row['cvar_95']:.3f} utility={row['utility']:.3f}")
        except Exception as exc:
            print(f"  [{universe}] {method} FAILED: {exc}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universes", nargs="+", default=["spy", "qqq", "slv", "spx"])
    ap.add_argument("--reports-dir", default="reports_real")
    args = ap.parse_args()

    all_rows = []
    for u in args.universes:
        all_rows.extend(run_universe(u))

    out = ROOT / args.reports_dir / "tables" / "delta_gamma_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
