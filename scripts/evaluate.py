"""Stage 3 — backtest all policies on held-out paths and write reports/tables/figures.

    python scripts/evaluate.py --config configs/experiment.yaml

Reads artifacts/dataset.pkl and artifacts/models.pkl (run build_dataset.py and
train.py first). Produces reports/final_report.md and the figure/table set.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))


import argparse
import pickle
from pathlib import Path

from ivsh.pipeline import evaluate_and_report, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", default="artifacts/dataset.pkl")
    ap.add_argument("--models", default="artifacts/models.pkl")
    ap.add_argument("--out", default="artifacts/results.pkl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.dataset, "rb") as fh:
        data = pickle.load(fh)["data"]
    with open(args.models, "rb") as fh:
        models = pickle.load(fh)["models"]

    res = evaluate_and_report(cfg, data, models)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"comparison": res["comparison"], "stats": res["stats"], "manifest": res["manifest"]}, fh)
    print(res["comparison"].to_string())
    print(f"\nresults summary -> {args.out};  reports in {cfg.reports_dir}/")


if __name__ == "__main__":
    main()
