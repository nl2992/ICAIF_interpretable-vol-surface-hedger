"""Stage 4 — surface the final report for an experiment.

    python scripts/make_report.py --experiment_id ivsh_demo

The markdown reports are produced by the evaluate stage; this command verifies
them, prints the headline comparison, and points at the artifacts.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))


import argparse
import json
import pickle
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment_id", default="ivsh_demo")
    ap.add_argument("--results", default="artifacts/results.pkl")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    reports = Path(args.reports_dir)
    manifest_path = reports / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(f"experiment_id : {manifest.get('experiment_id')}")
        print(f"dataset       : {manifest.get('dataset_version')}")
        print(f"model         : {manifest.get('model_version')}")
        print(f"split         : {manifest.get('split_id')}")

    if Path(args.results).exists():
        with open(args.results, "rb") as fh:
            res = pickle.load(fh)
        print("\nModel comparison (test set):")
        print(res["comparison"].to_string())

    produced = [
        reports / "final_report.md",
        reports / "prototype_audit_report.md",
        reports / "ablation_report.md",
    ]
    print("\nReports:")
    for p in produced:
        print(f"  {'OK ' if p.exists() else 'MISSING '}{p}")


if __name__ == "__main__":
    main()
