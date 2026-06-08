"""Build real-data episode banks once and pickle them for fast reuse.

The SVI surface fit over millions of quotes is the expensive step (~30-60 min);
the grid search must not pay it per config. This builds each universe's
``EpisodeBank`` once and pickles it to ``artifacts/bank_<name>.pkl``.

    python scripts/cache_banks.py \
        --universe spy="data/raw/spy/spy_eod_*.txt" \
        --universe qqq="data/raw/qqq/qqq_eod_*.txt" \
        --universe slv="data/raw/slv/slv_eod_*.txt" --thin-surface
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ivsh.data.clean import clean_option_panel
from ivsh.data.loaders import load_optionsdx, market_from_option_panel
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank, concat_banks

ARTIFACTS = _pl.Path(__file__).resolve().parents[1] / "artifacts"


def build_bank(
    globs,
    surface="svi",
    *,
    max_rel_spread=0.5,
    moneyness_low=0.80,
    moneyness_high=1.20,
    min_volume=1,
    max_calendar_gap_days=10,
):
    parts = []
    summaries = []
    for pat in globs:
        p = load_optionsdx(pat)
        c, summary = clean_option_panel(
            p,
            max_rel_spread=max_rel_spread,
            iv_bounds=(0.03, 1.5),
            moneyness_band=(moneyness_low, moneyness_high),
            otm_only=True,
            min_volume=min_volume,
        )
        parts.append(c)
        summaries.append(summary.table.assign(source=pat))
        print(f"    {pat}: {len(c):,} clean")
    clean = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    unique_dates = pd.Series(sorted(pd.unique(pd.to_datetime(clean["date"]))))
    segment_id = (unique_dates.diff().dt.days.fillna(0) > max_calendar_gap_days).cumsum()
    date_to_segment = dict(zip(unique_dates, segment_id))
    clean["_segment"] = pd.to_datetime(clean["date"]).map(date_to_segment)

    banks = []
    year_parts = []
    for seg, panel in clean.groupby("_segment", sort=True):
        panel = panel.drop(columns="_segment").reset_index(drop=True)
        dates = np.array(sorted(pd.unique(pd.to_datetime(panel["date"]))))
        if len(dates) <= EnvConfig().liab_tenor_days + 1:
            print(f"  skip segment {seg}: only {len(dates)} dates")
            continue
        market = market_from_option_panel(panel, surface_method=surface)
        seg_bank = build_episode_bank(market, EnvConfig())
        banks.append(seg_bank)
        year_parts.append(pd.to_datetime(dates[seg_bank.start_days]).year.to_numpy())
        print(f"  segment {seg}: {pd.Timestamp(dates[0]).date()} to "
              f"{pd.Timestamp(dates[-1]).date()}, "
              f"{seg_bank.n_episodes} episodes")
    if not banks:
        raise ValueError("no contiguous date segment is long enough to build episodes")
    bank = banks[0] if len(banks) == 1 else concat_banks(banks)
    years = np.concatenate(year_parts)
    print(f"  bank: {bank.n_episodes} episodes, years {years.min()}-{years.max()}")
    return bank, years, pd.concat(summaries, ignore_index=True)


def _parse_universe(spec):
    name, glob = spec.split("=", 1)
    return name, glob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", action="append", nargs="+", required=True,
                    help="NAME=GLOB [extra globs...]; repeatable")
    ap.add_argument("--surface", choices=["ols", "svi"], default="svi")
    ap.add_argument("--max-rel-spread", type=float, default=0.5)
    ap.add_argument("--moneyness-low", type=float, default=0.80)
    ap.add_argument("--moneyness-high", type=float, default=1.20)
    ap.add_argument("--min-volume", type=int, default=1)
    ap.add_argument("--max-calendar-gap-days", type=int, default=10)
    ap.add_argument("--thin-surface", action="store_true",
                    help="looser defaults for sparse chains such as SLV")
    args = ap.parse_args()
    if args.thin_surface:
        args.max_rel_spread = max(args.max_rel_spread, 0.8)
        args.moneyness_low = min(args.moneyness_low, 0.70)
        args.moneyness_high = max(args.moneyness_high, 1.30)
        args.min_volume = min(args.min_volume, 0)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for entry in args.universe:
        name, first = _parse_universe(entry[0])
        globs = [first] + entry[1:]
        print(f"[{name}]")
        bank, years, summary = build_bank(
            globs,
            args.surface,
            max_rel_spread=args.max_rel_spread,
            moneyness_low=args.moneyness_low,
            moneyness_high=args.moneyness_high,
            min_volume=args.min_volume,
            max_calendar_gap_days=args.max_calendar_gap_days,
        )
        out = ARTIFACTS / f"bank_{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump({"bank": bank, "years": years}, f)
        summary.to_csv(ARTIFACTS / f"cleaning_funnel_{name}.csv", index=False)
        print(f"  cached -> {out}")


if __name__ == "__main__":
    main()
