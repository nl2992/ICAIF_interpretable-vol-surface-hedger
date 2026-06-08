"""Annotate walk-forward CVaR tables with named stress regimes and plot them."""

from __future__ import annotations

import argparse
import pathlib as _pl
import sys as _sys

try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def regime_label(year: int) -> str:
    if year <= 2017:
        return "pre-2018 calm/mixed"
    if year == 2018:
        return "Volmageddon/Q4 selloff"
    if year == 2019:
        return "low-vol expansion"
    if year == 2020:
        return "COVID vol spike"
    if year == 2021:
        return "recovery"
    if year == 2022:
        return "rate-shock bear market"
    if year == 2023:
        return "normalisation"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", default="reports_real")
    ap.add_argument("--universe", nargs="+", default=["spy", "qqq"])
    args = ap.parse_args()

    root = _pl.Path(args.reports_dir)
    tables, figs = root / "tables", root / "figures"
    rows = []
    for i, u in enumerate(args.universe):
        path = tables / ("walkforward_cvar.csv" if i == 0 and u == "spy" else f"walkforward_cvar_{u}.csv")
        if not path.exists():
            alt = tables / f"walkforward_cvar_{u}.csv"
            path = alt if alt.exists() else path
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        df = pd.read_csv(path)
        df.insert(0, "universe", u)
        df["macro_regime"] = df["test_year"].map(regime_label)
        rows.append(df)

    if not rows:
        raise SystemExit("no walk-forward tables found")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(tables / "walkforward_stress_audit.csv", index=False)

    methods = [c for c in ["delta_vega", "prototype", "prototype_capped", "blackbox", "ppo"] if c in out.columns]
    fig, axes = plt.subplots(len(args.universe), 1, figsize=(9, 3.2 * len(args.universe)), squeeze=False)
    for ax, u in zip(axes[:, 0], args.universe):
        sub = out[out["universe"] == u]
        for m in methods:
            ax.plot(sub["test_year"], sub[m], marker="o", label=m)
        for y in (2020, 2022):
            if y in set(sub["test_year"]):
                ax.axvspan(y - 0.35, y + 0.35, color="#f0c36a", alpha=0.18)
        ax.set_title(f"{u.upper()} walk-forward CVaR95 by named regime")
        ax.set_ylabel("CVaR95")
        ax.set_yscale("log")
        ax.legend(fontsize=8, ncol=3)
    axes[-1, 0].set_xlabel("test year")
    fig.tight_layout()
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "walkforward_stress_audit.png", dpi=140)
    print(f"wrote {tables / 'walkforward_stress_audit.csv'}")


if __name__ == "__main__":
    main()
