"""Run the full pipeline (data -> train -> evaluate -> report) in one command.

    python scripts/run_experiment.py --config configs/experiment.yaml
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))


import argparse

from ivsh.pipeline import ExperimentConfig, load_config, run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--no-ablations", action="store_true", help="skip ablation runs")
    args = ap.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    if args.no_ablations:
        cfg.run_ablations = False

    res = run_experiment(cfg)
    comp = res["comparison"]
    print(comp.to_string())
    print(f"\nReports written to {cfg.reports_dir}/ (final_report.md, prototype_audit_report.md, ablation_report.md)")


if __name__ == "__main__":
    main()
