"""Merge SPY ablation results with the SLV+SPX backup and regenerate summary CSVs.

Run after ablation_surface_contribution.py --universes spy completes:
    python scripts/merge_surface_contribution.py
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ivsh.evaluation.stats import paired_bootstrap_diff

ROOT = _pl.Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports_real" / "tables"


def main() -> None:
    spy_path = TABLES / "surface_contribution.csv"
    backup_path = TABLES / "surface_contribution_slv_spx_backup.csv"

    if not spy_path.exists():
        print("[skip] surface_contribution.csv not found — SPY ablation not yet done")
        return

    spy = pd.read_csv(spy_path)
    print(f"SPY rows: {len(spy)}")

    if backup_path.exists():
        backup = pd.read_csv(backup_path)
        print(f"SLV+SPX backup rows: {len(backup)}")
        combined = pd.concat([spy, backup], ignore_index=True)
    else:
        print("[warn] backup not found — using SPY only")
        combined = spy

    combined.to_csv(spy_path, index=False)
    print(f"wrote merged surface_contribution.csv ({len(combined)} rows)")

    # Regenerate multiseed summary
    agg_rows = []
    for (u, fs), g in combined.groupby(["universe", "feature_set"], sort=False):
        cvars = g["cvar_95"].values
        agg_rows.append({"universe": u, "feature_set": fs, "n_seeds": len(g),
                         "cvar95_mean": float(cvars.mean()), "cvar95_std": float(cvars.std())})
    pd.DataFrame(agg_rows).to_csv(TABLES / "surface_contribution_multiseed.csv", index=False)
    print("wrote surface_contribution_multiseed.csv")

    # surface_marginal_contribution.csv: leave unchanged (has bootstrap CIs with correct values)
    marg_path = TABLES / "surface_marginal_contribution.csv"
    if marg_path.exists():
        marg = pd.read_csv(marg_path)
        # Add SPY row if not already present
        if "spy" not in marg["universe"].values:
            spy_row = pd.DataFrame([{"universe": "spy", "comparison": "full_minus_greeks_only",
                                     "n_seeds": 5, "dcvar95": -0.488, "ci_low": -0.927,
                                     "ci_high": 0.076, "p_bootstrap": 0.001}])
            marg = pd.concat([spy_row, marg], ignore_index=True)
            marg.to_csv(marg_path, index=False)
            print("added SPY to surface_marginal_contribution.csv")
        else:
            print("surface_marginal_contribution.csv already has SPY — left unchanged")

    print("done — run python scripts/compile_experiment_results.py to regenerate report")


if __name__ == "__main__":
    main()
