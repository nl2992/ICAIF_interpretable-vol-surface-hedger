"""Extra real-data analyses: multi-seed robustness and walk-forward by year.

Builds the surface and episode bank ONCE from the OptionsDX files, then:
  (1) multi-seed: re-fits the learned hedgers across several seeds on the standard
      chronological split and reports CVaR95 mean +/- std per policy;
  (2) walk-forward: for each test year, trains on all prior years and evaluates on
      that year, giving a per-regime CVaR95 track.

    python scripts/run_real_analysis.py --data "data/raw/spy/spy_eod_20*.txt"
"""

import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ivsh.data.clean import clean_option_panel
from ivsh.data.loaders import load_optionsdx, market_from_option_panel
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank
from ivsh.evaluation.backtest import run_baseline, run_policy
from ivsh.training.objective import cvar_from_pnl
from ivsh.training.train import TrainConfig, fit_blackbox, fit_prototype, make_standardizer
from ivsh.utils.splits import subset

SEEDS = (7, 13, 23, 42, 2025)
METHODS = ("delta", "delta_vega", "blackbox", "prototype")


def _build_bank(data_globs, surface):
    parts = []
    for pat in data_globs:
        p = load_optionsdx(pat)
        c, _ = clean_option_panel(p, max_rel_spread=0.5, iv_bounds=(0.03, 1.5),
                                  moneyness_band=(0.80, 1.20), otm_only=True, min_volume=1)
        parts.append(c)
        print(f"  {pat}: {len(c):,} clean")
    clean = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    market = market_from_option_panel(clean, surface_method=surface)
    bank = build_episode_bank(market, EnvConfig())
    dates = np.array(sorted(pd.unique(pd.to_datetime(clean["date"]))))
    years = pd.to_datetime(dates[bank.start_days]).year.to_numpy()
    print(f"bank: {bank.n_episodes} episodes, {bank.n_days if hasattr(bank,'n_days') else '?'} ; "
          f"years {years.min()}-{years.max()}")
    return bank, years


def _fit_eval(trb, vlb, teb, seed, max_iter, anchor=True, rscale=1.5):
    scaler = make_standardizer(trb)
    pcfg = TrainConfig(n_prototypes=8, l2=1e-3, max_iter=max_iter, anchor=anchor, action_scale=rscale, seed=seed)
    bcfg = TrainConfig(hidden=16, l2=3e-2, max_iter=max_iter, anchor=anchor, action_scale=rscale, seed=seed)
    proto, _, _ = fit_prototype(trb, scaler, pcfg, val_bank=vlb)
    bb, _ = fit_blackbox(trb, scaler, bcfg, val_bank=vlb)
    out = {}
    for m in ("delta", "delta_vega"):
        out[m] = cvar_from_pnl(run_baseline(teb, m)["pnl"])
    out["blackbox"] = cvar_from_pnl(run_policy(bb, teb, scaler, anchor=anchor)["pnl"])
    out["prototype"] = cvar_from_pnl(run_policy(proto, teb, scaler, anchor=anchor)["pnl"])
    return out


def multiseed(bank, max_iter, tables, figs):
    from ivsh.utils.splits import chronological_split

    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    rows = []
    for s in SEEDS:
        r = _fit_eval(trb, vlb, teb, s, max_iter)
        r["seed"] = s
        rows.append(r)
        print(f"  seed {s}: " + ", ".join(f"{m}={r[m]:.3f}" for m in METHODS))
    df = pd.DataFrame(rows).set_index("seed")
    summ = pd.DataFrame({"cvar95_mean": df[list(METHODS)].mean(), "cvar95_std": df[list(METHODS)].std()})
    df.to_csv(tables / "multiseed_cvar_byseed.csv")
    summ.to_csv(tables / "multiseed_cvar.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(summ.index, summ["cvar95_mean"], yerr=summ["cvar95_std"], capsize=5,
           color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"], edgecolor="k", linewidth=0.5)
    ax.set_ylabel("CVaR95 (mean +/- std over 5 seeds)")
    ax.set_title("Real SPY 2010-2023: multi-seed robustness")
    fig.tight_layout(); fig.savefig(figs / "multiseed_cvar.png", dpi=130); plt.close(fig)
    print(summ.round(3).to_string())
    return summ


def walkforward(bank, years, max_iter, tables, figs, min_train=80):
    rows = []
    test_years = [y for y in range(int(years.min()) + 4, int(years.max()) + 1)]
    horizon = bank.horizon
    for ty in test_years:
        test_idx = np.where(years == ty)[0]
        if len(test_idx) < 20:
            continue
        test_start = bank.start_days[test_idx].min()
        train_idx = np.where((years < ty) & (bank.start_days < test_start - horizon))[0]
        if len(train_idx) < min_train:
            continue
        # last 15% of train (by start day) as validation
        order = train_idx[np.argsort(bank.start_days[train_idx])]
        cut = int(len(order) * 0.85)
        trb = subset(bank, order[:cut]); vlb = subset(bank, order[cut:]); teb = subset(bank, test_idx)
        r = _fit_eval(trb, vlb, teb, seed=7, max_iter=max_iter)
        r["test_year"] = ty; r["n_test"] = len(test_idx); r["n_train"] = len(train_idx)
        rows.append(r)
        print(f"  test {ty} (n={len(test_idx)}): " + ", ".join(f"{m}={r[m]:.3f}" for m in METHODS))
    if not rows:
        print("  (no walk-forward folds with enough history)")
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("test_year")
    df.to_csv(tables / "walkforward_cvar.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    colors = {"delta": "#4c72b0", "delta_vega": "#55a868", "blackbox": "#c44e52", "prototype": "#8172b3"}
    for m in METHODS:
        ax.plot(df.index, df[m], marker="o", label=m, color=colors[m])
    ax.set_ylabel("CVaR95 tail loss"); ax.set_xlabel("test year (train = all prior years)")
    ax.set_title("Real SPY walk-forward: CVaR95 by test year")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(figs / "walkforward_cvar.png", dpi=130); plt.close(fig)
    print(df.round(3).to_string())
    # win-rate summary
    wins = int((df["prototype"] <= df["delta_vega"]).sum())
    print(f"prototype <= delta_vega in {wins}/{len(df)} test years")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--surface", choices=["ols", "svi"], default="svi")
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--reports-dir", default="reports_real")
    args = ap.parse_args()

    tables = _pl.Path(args.reports_dir) / "tables"
    figs = _pl.Path(args.reports_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    print("building surface + bank (once) ...")
    bank, years = _build_bank(args.data, args.surface)
    print("\n== multi-seed robustness ==")
    multiseed(bank, args.max_iter, tables, figs)
    print("\n== walk-forward by year ==")
    walkforward(bank, years, args.max_iter, tables, figs)
    print(f"\nAnalysis artefacts written to {args.reports_dir}/")


if __name__ == "__main__":
    main()
