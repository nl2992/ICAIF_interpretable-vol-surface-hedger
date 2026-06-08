"""Real-data robustness analyses: multi-universe significance, deep-RL (PPO/SAC)
stress comparison, and a volatility-capped walk-forward.

Three studies, all on real extracted OptionsDX chains:

  (1) **headline / multi-universe** — on each universe's chronological test split,
      score delta / delta-vega / black-box-MLP / prototype / PPO / SAC, write a
      per-universe comparison + paired-bootstrap significance (prototype vs each),
      and combine across universes with Stouffer's method.
  (2) **multi-seed** — refit the learned hedgers across seeds on the primary
      universe's standard split; report CVaR95 mean +/- std per policy (shows the
      deep hedgers' seed instability).
  (3) **walk-forward** — train on all prior years, test on each later year, for the
      prototype, its **volatility-capped** variant (the COVID-2020 fix), the
      baselines and PPO.

Examples
--------
    # smoke test (tiny budgets, one universe, ~minutes)
    python scripts/run_real_analysis.py --quick \
        --universe spy="data/raw/spy/spy_eod_2018*.txt" "data/raw/spy/spy_eod_2019*.txt" \
                       "data/raw/spy/spy_eod_2020*.txt"

    # full run (both universes)
    python scripts/run_real_analysis.py \
        --universe spy="data/raw/spy/spy_eod_*.txt" \
        --universe qqq="data/raw/qqq/qqq_eod_*.txt"
"""

import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")  # repo path / output contain non-ASCII
except Exception:
    pass

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ivsh.data.clean import clean_option_panel
from ivsh.data.loaders import load_optionsdx, market_from_option_panel
from ivsh.envs.hedging_env import EnvConfig, build_episode_bank, concat_banks
from ivsh.evaluation.backtest import run_baseline, run_policy
from ivsh.evaluation.metrics import compute_metrics
from ivsh.evaluation.stats import paired_bootstrap_diff, stouffer_combine, wilcoxon_pnl
from ivsh.training.objective import cvar_from_pnl
from ivsh.training.train import (
    TrainConfig,
    fit_blackbox,
    fit_prototype,
    make_standardizer,
    realized_vol_scale,
)
from ivsh.utils.splits import chronological_split, subset

from ivsh.viz import METHOD_COLORS as COLORS, apply_theme

apply_theme()
SEEDS = (7, 13, 23, 42, 2025)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _build_bank(data_globs, surface):
    parts = []
    for pat in data_globs:
        p = load_optionsdx(pat)
        c, _ = clean_option_panel(p, max_rel_spread=0.5, iv_bounds=(0.03, 1.5),
                                  moneyness_band=(0.80, 1.20), otm_only=True, min_volume=1)
        parts.append(c)
        print(f"    {pat}: {len(c):,} clean")
    clean = pd.concat(parts, ignore_index=True).sort_values("date").reset_index(drop=True)
    unique_dates = pd.Series(sorted(pd.unique(pd.to_datetime(clean["date"]))))
    segment_id = (unique_dates.diff().dt.days.fillna(0) > 10).cumsum()
    date_to_segment = dict(zip(unique_dates, segment_id))
    clean["_segment"] = pd.to_datetime(clean["date"]).map(date_to_segment)
    banks, year_parts = [], []
    for seg, panel in clean.groupby("_segment", sort=True):
        panel = panel.drop(columns="_segment").reset_index(drop=True)
        dates = np.array(sorted(pd.unique(pd.to_datetime(panel["date"]))))
        if len(dates) <= EnvConfig().liab_tenor_days + 1:
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
    return bank, years


# --------------------------------------------------------------------------- #
# Model fitting / scoring
# --------------------------------------------------------------------------- #
def _train_cfg(seed, max_iter, anchor, rscale, hidden=False):
    if hidden:
        return TrainConfig(hidden=16, l2=3e-2, max_iter=max_iter, anchor=anchor,
                           action_scale=rscale, seed=seed)
    return TrainConfig(n_prototypes=8, l2=1e-3, max_iter=max_iter, anchor=anchor,
                       action_scale=rscale, seed=seed)


def fit_all(trb, vlb, teb, seed, methods, *, max_iter=250, anchor=True, rscale=1.5,
            rl_timesteps=60_000, vol_floor=0.25):
    """Fit/score the requested methods; return ``{method: pnl_vector_on_test}``."""
    scaler = make_standardizer(trb)
    out = {}

    if "delta" in methods:
        out["delta"] = run_baseline(teb, "delta")["pnl"]
    if "delta_vega" in methods:
        out["delta_vega"] = run_baseline(teb, "delta_vega")["pnl"]
    if "blackbox" in methods:
        bb, _ = fit_blackbox(trb, scaler, _train_cfg(seed, max_iter, anchor, rscale, hidden=True), val_bank=vlb)
        out["blackbox"] = run_policy(bb, teb, scaler, anchor=anchor)["pnl"]
    if "prototype" in methods:
        proto, _, _ = fit_prototype(trb, scaler, _train_cfg(seed, max_iter, anchor, rscale), val_bank=vlb)
        out["prototype"] = run_policy(proto, teb, scaler, anchor=anchor)["pnl"]
    if "prototype_capped" in methods:
        sc_tr, ref = realized_vol_scale(trb, floor=vol_floor)
        sc_vl, _ = realized_vol_scale(vlb, ref=ref, floor=vol_floor) if vlb.n_episodes else (None, ref)
        sc_te, _ = realized_vol_scale(teb, ref=ref, floor=vol_floor)
        pc, _, _ = fit_prototype(trb, scaler, _train_cfg(seed, max_iter, anchor, rscale),
                                 val_bank=vlb, residual_scale=sc_tr, val_residual_scale=sc_vl)
        out["prototype_capped"] = run_policy(pc, teb, scaler, anchor=anchor, residual_scale=sc_te)["pnl"]
    rl_algos = [m for m in ("ppo", "sac") if m in methods]
    if rl_algos:
        from ivsh.models.deep_rl import RLConfig, evaluate_sb3, train_sb3

        for algo in rl_algos:
            model = train_sb3(trb, scaler, RLConfig(algo=algo, total_timesteps=rl_timesteps,
                                                    action_scale=rscale, seed=seed))
            out[algo] = evaluate_sb3(model, teb, scaler, action_scale=rscale)["pnl"]
    return out


# --------------------------------------------------------------------------- #
# (1) Headline + multi-universe significance
# --------------------------------------------------------------------------- #
def headline(universes, methods, tables, figs, *, max_iter, rl_timesteps, seed=7):
    print("\n== headline + multi-universe significance ==")
    per_universe_pnl = {}
    cmp_rows = []
    for uname, (bank, _years) in universes.items():
        sp = chronological_split(bank)
        trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
        print(f"  [{uname}] train {trb.n_episodes} / val {vlb.n_episodes} / test {teb.n_episodes}")
        pnl = fit_all(trb, vlb, teb, seed, methods, max_iter=max_iter, rl_timesteps=rl_timesteps)
        per_universe_pnl[uname] = pnl
        for m, v in pnl.items():
            row = {"universe": uname, "method": m}
            row.update(compute_metrics(v))
            cmp_rows.append(row)
            print(f"    {m:18s} cvar95={cvar_from_pnl(v):.3f} mean={v.mean():.3f}")
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(tables / "multiverse_comparison.csv", index=False)

    # Significance: prototype vs each other method, per universe + Stouffer combined.
    others = [m for m in methods if m != "prototype"]
    sig_rows = []
    for m in others:
        per = []
        for uname, pnl in per_universe_pnl.items():
            if "prototype" not in pnl or m not in pnl:
                continue
            bs = paired_bootstrap_diff(pnl["prototype"], pnl[m], stat="cvar")
            wl = wilcoxon_pnl(pnl["prototype"], pnl[m])
            per.append(bs)
            sig_rows.append({"comparison": f"prototype - {m}", "universe": uname,
                             "dcvar95": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
                             "p_bootstrap": bs["p_two_sided"], "p_wilcoxon": wl["pvalue"]})
        if len(per) > 1:
            comb = stouffer_combine(per)
            sig_rows.append({"comparison": f"prototype - {m}", "universe": "COMBINED",
                             "dcvar95": comb["mean_diff"], "ci_low": np.nan, "ci_high": np.nan,
                             "p_bootstrap": comb["p_two_sided"], "p_wilcoxon": np.nan})
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(tables / "multiverse_significance.csv", index=False)
    print(sig.round(4).to_string(index=False))

    # Grouped bar: CVaR95 by method, grouped by universe.
    unames = list(universes)
    ms = [m for m in methods if any(m in per_universe_pnl[u] for u in unames)]
    x = np.arange(len(unames))
    w = 0.8 / max(len(ms), 1)
    fig, ax = plt.subplots(figsize=(2.2 * len(unames) + 4, 4.5))
    for i, m in enumerate(ms):
        vals = [cvar_from_pnl(per_universe_pnl[u][m]) if m in per_universe_pnl[u] else np.nan for u in unames]
        ax.bar(x + i * w, vals, w, label=m, color=COLORS.get(m, None), edgecolor="k", linewidth=0.4)
    ax.set_xticks(x + w * (len(ms) - 1) / 2)
    ax.set_xticklabels([u.upper() for u in unames])
    ax.set_ylabel("CVaR95 tail loss (test split)")
    ax.set_title("Multi-universe hedging comparison")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(figs / "multiverse_cvar.png", dpi=130); plt.close(fig)
    return cmp, sig


# --------------------------------------------------------------------------- #
# (2) Multi-seed robustness (primary universe)
# --------------------------------------------------------------------------- #
def multiseed(bank, methods, tables, figs, *, max_iter, rl_timesteps):
    print("\n== multi-seed robustness (primary universe) ==")
    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    rows = []
    for s in SEEDS:
        pnl = fit_all(trb, vlb, teb, s, methods, max_iter=max_iter, rl_timesteps=rl_timesteps)
        r = {m: cvar_from_pnl(v) for m, v in pnl.items()}
        r["seed"] = s
        rows.append(r)
        print("  seed %4d: " % s + ", ".join(f"{m}={r[m]:.3f}" for m in pnl))
    df = pd.DataFrame(rows).set_index("seed")
    cols = [m for m in methods if m in df.columns]
    summ = pd.DataFrame({"cvar95_mean": df[cols].mean(), "cvar95_std": df[cols].std()})
    df.to_csv(tables / "multiseed_cvar_byseed.csv")
    summ.to_csv(tables / "multiseed_cvar.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(summ.index, summ["cvar95_mean"], yerr=summ["cvar95_std"], capsize=5,
           color=[COLORS.get(m) for m in summ.index], edgecolor="k", linewidth=0.5)
    ax.set_ylabel("CVaR95 (mean +/- std over %d seeds)" % len(SEEDS))
    ax.set_title("Multi-seed robustness")
    fig.tight_layout(); fig.savefig(figs / "multiseed_cvar.png", dpi=130); plt.close(fig)
    print(summ.round(3).to_string())
    return summ


# --------------------------------------------------------------------------- #
# (3) Walk-forward with the volatility-capped fix
# --------------------------------------------------------------------------- #
def walkforward(bank, years, methods, tables, figs, *, max_iter, rl_timesteps, suffix="",
                min_train=80):
    print(f"\n== walk-forward by year{(' [' + suffix + ']') if suffix else ''} ==")
    rows = []
    horizon = bank.horizon
    test_years = [y for y in range(int(years.min()) + 4, int(years.max()) + 1)]
    for ty in test_years:
        test_idx = np.where(years == ty)[0]
        if len(test_idx) < 20:
            continue
        test_start = bank.start_days[test_idx].min()
        train_idx = np.where((years < ty) & (bank.start_days < test_start - horizon))[0]
        if len(train_idx) < min_train:
            continue
        order = train_idx[np.argsort(bank.start_days[train_idx])]
        cut = int(len(order) * 0.85)
        trb, vlb, teb = subset(bank, order[:cut]), subset(bank, order[cut:]), subset(bank, test_idx)
        pnl = fit_all(trb, vlb, teb, 7, methods, max_iter=max_iter, rl_timesteps=rl_timesteps)
        r = {m: cvar_from_pnl(v) for m, v in pnl.items()}
        r["test_year"] = ty; r["n_test"] = len(test_idx); r["n_train"] = len(train_idx)
        rows.append(r)
        print(f"  test {ty} (n={len(test_idx)}): " + ", ".join(f"{m}={r[m]:.3f}" for m in pnl))
    if not rows:
        print("  (no folds with enough history)")
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("test_year")
    df.to_csv(tables / f"walkforward_cvar{suffix}.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for m in [c for c in methods if c in df.columns]:
        ax.plot(df.index, df[m], marker="o", label=m, color=COLORS.get(m))
    ax.set_ylabel("CVaR95 tail loss"); ax.set_xlabel("test year (train = all prior years)")
    ax.set_title(f"Walk-forward CVaR95 by year{(' — ' + suffix) if suffix else ''}")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figs / f"walkforward_cvar{suffix}.png", dpi=130); plt.close(fig)
    print(df.round(3).to_string())
    if "prototype" in df and "prototype_capped" in df:
        worst = df["prototype"].idxmax()
        print(f"  worst prototype fold = {worst}: uncapped {df.loc[worst,'prototype']:.2f} "
              f"-> capped {df.loc[worst,'prototype_capped']:.2f}")
    return df


# --------------------------------------------------------------------------- #
def _parse_universe(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError("use name=glob")
    name, glob = spec.split("=", 1)
    return name, glob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", action="append", nargs="+", default=[],
                    help="NAME=GLOB [extra globs...]; repeatable (first = primary)")
    ap.add_argument("--data", nargs="+", help="(back-compat) globs for a single 'spy' universe")
    ap.add_argument("--surface", choices=["ols", "svi"], default="svi")
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--rl-timesteps", type=int, default=60_000)
    ap.add_argument("--reports-dir", default="reports_real")
    ap.add_argument("--quick", action="store_true", help="smoke mode: tiny budgets")
    ap.add_argument("--skip", nargs="*", default=[], choices=["headline", "multiseed", "walkforward"])
    args = ap.parse_args()

    if args.quick:
        args.max_iter = min(args.max_iter, 60)
        args.rl_timesteps = min(args.rl_timesteps, 3_000)

    # Resolve universes: each --universe entry is [name=glob, glob2, ...].
    universe_globs: dict[str, list[str]] = {}
    for entry in args.universe:
        name, first = _parse_universe(entry[0])
        universe_globs[name] = [first] + entry[1:]
    if args.data:
        universe_globs.setdefault("spy", args.data)
    if not universe_globs:
        ap.error("provide at least one --universe NAME=GLOB (or --data)")

    tables = _pl.Path(args.reports_dir) / "tables"
    figs = _pl.Path(args.reports_dir) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    print("building surfaces + banks ...")
    universes = {}
    for name, globs in universe_globs.items():
        print(f"  [{name}]")
        universes[name] = _build_bank(globs, args.surface)
    primary = next(iter(universes))

    headline_methods = ["delta", "delta_vega", "blackbox", "prototype", "ppo", "sac"]
    multiseed_methods = ["delta", "delta_vega", "blackbox", "prototype", "ppo"]
    wf_methods = ["delta", "delta_vega", "blackbox", "prototype", "prototype_capped", "ppo"]

    if "headline" not in args.skip:
        headline(universes, headline_methods, tables, figs,
                 max_iter=args.max_iter, rl_timesteps=args.rl_timesteps)
    if "multiseed" not in args.skip:
        multiseed(universes[primary][0], multiseed_methods, tables, figs,
                  max_iter=args.max_iter, rl_timesteps=args.rl_timesteps)
    if "walkforward" not in args.skip:
        for name, (bank, years) in universes.items():
            suffix = "" if name == primary else f"_{name}"
            walkforward(bank, years, wf_methods, tables, figs,
                        max_iter=args.max_iter, rl_timesteps=args.rl_timesteps, suffix=suffix)

    print(f"\nAnalysis artefacts written to {args.reports_dir}/")


if __name__ == "__main__":
    main()
