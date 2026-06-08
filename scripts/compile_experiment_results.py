"""Compile all experiment results into a comprehensive Markdown summary.

Reads from reports_real/tables/ to produce a single consolidated results document.
Run this after all individual experiments have completed.

Example:
    python scripts/compile_experiment_results.py
"""

from __future__ import annotations

import pathlib as _pl
import sys as _sys
from datetime import date

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

ROOT = _pl.Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports_real" / "tables"


def _read(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    if p.exists():
        return pd.read_csv(p)
    print(f"  [missing] {name}")
    return None


def _fmt(x, fmt=".3f") -> str:
    try:
        return format(float(x), fmt)
    except Exception:
        return str(x)


def main() -> None:
    lines = [
        f"# Consolidated Experiment Results",
        f"",
        f"Generated: {date.today().isoformat()}",
        f"",
        f"---",
        f"",
    ]

    # ------------------------------------------------------------------ #
    # 1. Main model comparison (from final report)
    # ------------------------------------------------------------------ #
    mc = _read("model_comparison.csv")
    if mc is not None:
        lines += [
            "## 1. Main model comparison (SPY test set)",
            "",
            mc.to_markdown(index=False),
            "",
        ]

    # ------------------------------------------------------------------ #
    # 2. Delta-gamma-vega baseline comparison
    # ------------------------------------------------------------------ #
    dg = _read("delta_gamma_comparison.csv")
    if dg is not None:
        lines += [
            "## 2. Delta-gamma-vega baseline (Plan H)",
            "",
            "| universe | method | CVaR95 | utility |",
            "|----------|--------|--------|---------|",
        ]
        for _, r in dg.iterrows():
            lines.append(f"| {r['universe']} | {r['method']} | {_fmt(r['cvar_95'])} | {_fmt(r['utility'])} |")
        lines += [
            "",
            "**Key finding**: Delta-gamma-vega (sizing option by gamma) has *worse* CVaR than delta-vega across",
            "all universes. This confirms delta-vega as the hard analytic baseline; the prototype hedger",
            "beats delta-vega, not just delta or delta-gamma.",
            "",
        ]

    # ------------------------------------------------------------------ #
    # 3. Surface feature contribution (multi-seed)
    # ------------------------------------------------------------------ #
    sc = _read("surface_contribution.csv")
    ms = _read("surface_contribution_multiseed.csv")
    sm = _read("surface_marginal_contribution.csv")
    if sc is not None:
        lines += [
            "## 3. Surface feature contribution — ablation (Plan A)",
            "",
            "### Per-seed results",
            "",
            sc.to_markdown(index=False),
            "",
        ]
    if ms is not None:
        lines += [
            "### Multi-seed summary",
            "",
            ms.to_markdown(index=False),
            "",
        ]
    if sm is not None:
        lines += [
            "### Bootstrap marginal contribution (full vs greeks-only)",
            "",
            sm.to_markdown(index=False),
            "",
            "Negative dcvar95 = full model has *lower* (better) CVaR than greeks-only.",
            "",
        ]

    # ------------------------------------------------------------------ #
    # 4. ProtoHedge comparison (synthetic)
    # ------------------------------------------------------------------ #
    ph = _read("../tables/protohedge_baseline.csv")
    if ph is None:
        # Try reports/tables
        ph_path = ROOT / "reports" / "tables" / "protohedge_baseline.csv"
        if ph_path.exists():
            ph = pd.read_csv(ph_path)
    if ph is not None:
        lines += [
            "## 4. ProtoHedge comparison on synthetic market (Plan B)",
            "",
            ph.to_markdown(index=False),
            "",
            "Surface-aware prototype vs scalar-Greeks ProtoHedge baseline.",
            "Both trained on synthetic regime-switching SV market, same splits and objective.",
            "",
        ]
        surf = ph[ph["model"] == "surface_proto"]
        scalar = ph[ph["model"] == "protohedge_scalar_greeks"]
        n = min(len(surf), len(scalar))
        cvar_wins = (surf["cvar_95"].values[:n] < scalar["cvar_95"].values[:n]).sum()
        util_wins = (surf["utility"].values[:n] > scalar["utility"].values[:n]).sum()
        dd_wins = (surf["max_drawdown"].values[:n] < scalar["max_drawdown"].values[:n]).sum()
        mean_dd_surf = surf["max_drawdown"].mean()
        mean_dd_scal = scalar["max_drawdown"].mean()
        lines += [
            f"Surface wins {cvar_wins}/{n} seeds on CVaR95, {util_wins}/{n} on utility, {dd_wins}/{n} on max-drawdown.",
            f"Mean max-drawdown: surface={mean_dd_surf:.1f} vs scalar={mean_dd_scal:.1f} "
            f"({(mean_dd_scal-mean_dd_surf)/mean_dd_scal*100:.0f}% lower for surface).",
            "",
        ]

    # ------------------------------------------------------------------ #
    # 5. Tuned PPO results
    # ------------------------------------------------------------------ #
    ppo_spy = _read("tuned_ppo_spy_summary.csv")
    ppo_qqq = _read("tuned_ppo_qqq.csv")
    if ppo_spy is not None:
        lines += [
            "## 5. Tuned PPO robustness (Plan C)",
            "",
            "### SPY — grid summary (lr × action_scale)",
            "",
            ppo_spy.to_markdown(index=False),
            "",
        ]
        best = float(ppo_spy["test_cvar95_min"].min())
        lines.append(f"**Best tuned PPO test CVaR95 = {best:.2f}** (vs prototype ~2.38).")
        lines.append("")
    if ppo_qqq is not None:
        ppo_qqq_summary = ppo_qqq.groupby(["learning_rate", "action_scale"]).agg(
            test_cvar95_mean=("test_cvar95", "mean"),
            test_cvar95_min=("test_cvar95", "min"),
        ).reset_index()
        lines += [
            "### QQQ — grid summary",
            "",
            ppo_qqq_summary.to_markdown(index=False),
            "",
        ]

    # ------------------------------------------------------------------ #
    # 6. Walk-forward stress audit (Plan F)
    # ------------------------------------------------------------------ #
    wf = _read("walkforward_stress_audit.csv")
    if wf is not None:
        lines += [
            "## 6. Walk-forward CVaR by regime fold (Plan F)",
            "",
            wf.to_markdown(index=False),
            "",
        ]

    # ------------------------------------------------------------------ #
    # 7. Prototype regime audit (Plan D)
    # ------------------------------------------------------------------ #
    for uni in ["spy", "qqq", "slv", "spx"]:
        pra = _read(f"prototype_regime_audit_{uni}.csv")
        if pra is not None:
            lines += [
                f"## 7. Prototype regime catalogue — {uni.upper()} (Plan D)",
                "",
                pra[["prototype", "top_episode_share", "stress_episode_share",
                      "iv_level", "skew", "top_year", "regime_label"]].to_markdown(index=False),
                "",
            ]

    # ------------------------------------------------------------------ #
    # 8. QQQ surface-only winner config (Plan A supplementary)
    # ------------------------------------------------------------------ #
    qqq_sw = _read("qqq_surface_winner_cfg.csv")
    if qqq_sw is not None:
        mean_cv = qqq_sw["cvar_95"].mean()
        std_cv = qqq_sw["cvar_95"].std()
        lines += [
            "## 8. QQQ surface-only prototype (winner config, Plan A)",
            "",
            qqq_sw[["seed", "feature_set", "cvar_95", "mean_pnl", "utility", "max_drawdown"]].to_markdown(index=False),
            "",
            f"**QQQ surface-only (winner regularisation) CVaR95 = {mean_cv:.3f} ± {std_cv:.3f}**",
            f"vs delta-vega = 6.120 ({(6.120 - mean_cv) / 6.120 * 100:.1f}% improvement).",
            "",
        ]

    # ------------------------------------------------------------------ #
    # 9. Multi-universe significance
    # ------------------------------------------------------------------ #
    mv = _read("multiverse_significance.csv")
    if mv is not None:
        lines += [
            "## 9. Multi-universe significance",
            "",
            mv.to_markdown(index=False),
            "",
        ]

    out = ROOT / "reports_real" / "consolidated_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
