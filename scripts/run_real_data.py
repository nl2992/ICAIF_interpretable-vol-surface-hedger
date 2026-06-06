"""Run the hedging study on real OptionsDX option data.

    python scripts/run_real_data.py \
        --data "data/raw/spy/spy_eod_2018*.txt" "data/raw/spy/spy_eod_2019*.txt" \
               "data/raw/spy/spy_eod_2020*.txt" \
        --reports-dir reports_real --surface svi

Loads OptionsDX files (call+put wide -> long), cleans to a clean OTM smile,
fits the (SVI) surface, builds chronological train/val/test banks, trains the
prototype and black-box hedgers, and writes the report set with calendar-date
prototype annotations.
"""

import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))

import argparse

from ivsh.data.clean import clean_option_panel
from ivsh.data.loaders import load_optionsdx
from ivsh.envs.hedging_env import EnvConfig
from ivsh.pipeline import ExperimentConfig, run_experiment_real
from ivsh.training.train import TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True, help="OptionsDX file(s)/glob(s)")
    ap.add_argument("--experiment-id", default="spy_real")
    ap.add_argument("--reports-dir", default="reports_real")
    ap.add_argument("--checkpoints-dir", default="checkpoints")
    ap.add_argument("--surface", choices=["ols", "svi"], default="svi")
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--div", type=float, default=0.0)
    ap.add_argument("--n-prototypes", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument(
        "--no-anchor",
        action="store_true",
        help="learn absolute holdings (default: residual on top of delta-vega)",
    )
    ap.add_argument("--residual-scale", type=float, default=1.5)
    args = ap.parse_args()

    import pandas as pd

    # Clean each input pattern (e.g. one year) separately to bound peak memory,
    # then concatenate the (small) cleaned panels.
    parts = []
    summaries = []
    for pat in args.data:
        panel = load_optionsdx(pat)
        c, summ = clean_option_panel(
            panel,
            max_rel_spread=0.5,
            iv_bounds=(0.03, 1.5),
            moneyness_band=(0.80, 1.20),
            otm_only=True,
            min_volume=1,
        )
        parts.append(c)
        summaries.append(summ.table)
        print(f"  {pat}: {len(panel):,} raw -> {len(c):,} clean")
    clean = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    # aggregate cleaning funnel across inputs
    funnel = (
        pd.concat(summaries).groupby("filter", sort=False)["removed"].sum().reset_index()
    )
    print(f"total: {len(clean):,} clean OTM quotes; {clean['date'].nunique()} dates "
          f"({clean['date'].min().date()} -> {clean['date'].max().date()})")

    anchor = not args.no_anchor
    rscale = args.residual_scale
    cfg = ExperimentConfig(
        experiment_id=args.experiment_id,
        env=EnvConfig(),
        proto_train=TrainConfig(
            n_prototypes=args.n_prototypes, l2=1e-3, max_iter=args.max_iter,
            anchor=anchor, action_scale=rscale,
        ),
        bb_train=TrainConfig(
            hidden=16, l2=3e-2, max_iter=args.max_iter, anchor=anchor, action_scale=rscale,
        ),
        run_ablations=not args.no_ablations,
        reports_dir=args.reports_dir,
        checkpoints_dir=args.checkpoints_dir,
    )
    print(f"anchor (residual on delta-vega): {anchor}; residual scale: {rscale}")
    res = run_experiment_real(cfg, clean, rate=args.rate, div=args.div, surface_method=args.surface)
    funnel.to_csv(_pl.Path(args.reports_dir) / "tables" / "cleaning_funnel.csv", index=False)
    print(res["comparison"].to_string())
    print(f"\nReports written to {args.reports_dir}/")


if __name__ == "__main__":
    main()
