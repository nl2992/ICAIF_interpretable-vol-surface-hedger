"""Build the SPY + QQQ episode banks once and pickle them for fast reuse.

The SVI surface fit over millions of quotes is the expensive step (~30-60 min);
the grid search must not pay it per config. This builds each universe's
``EpisodeBank`` once and pickles it to ``artifacts/bank_<name>.pkl``.

    python scripts/cache_banks.py \
        --universe spy="data/raw/spy/spy_eod_*.txt" \
        --universe qqq="data/raw/qqq/qqq_eod_*.txt"
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
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank

ARTIFACTS = _pl.Path(__file__).resolve().parents[1] / "artifacts"


def build_bank(globs, surface="svi"):
    parts = []
    for pat in globs:
        p = load_optionsdx(pat)
        c, _ = clean_option_panel(p, max_rel_spread=0.5, iv_bounds=(0.03, 1.5),
                                  moneyness_band=(0.80, 1.20), otm_only=True, min_volume=1)
        parts.append(c)
        print(f"    {pat}: {len(c):,} clean")
    clean = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    market = market_from_option_panel(clean, surface_method=surface)
    bank = build_episode_bank(market, EnvConfig())
    dates = np.array(sorted(pd.unique(pd.to_datetime(clean["date"]))))
    years = pd.to_datetime(dates[bank.start_days]).year.to_numpy()
    print(f"  bank: {bank.n_episodes} episodes, years {years.min()}-{years.max()}")
    return bank, years


def _parse_universe(spec):
    name, glob = spec.split("=", 1)
    return name, glob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", action="append", nargs="+", required=True,
                    help="NAME=GLOB [extra globs...]; repeatable")
    ap.add_argument("--surface", choices=["ols", "svi"], default="svi")
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for entry in args.universe:
        name, first = _parse_universe(entry[0])
        globs = [first] + entry[1:]
        print(f"[{name}]")
        bank, years = build_bank(globs, args.surface)
        out = ARTIFACTS / f"bank_{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump({"bank": bank, "years": years}, f)
        print(f"  cached -> {out}")


if __name__ == "__main__":
    main()
