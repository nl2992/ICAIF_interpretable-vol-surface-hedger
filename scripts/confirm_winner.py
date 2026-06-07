"""Confirm the val-selected tail-weighted prototype on TEST for both universes.

The grid (scripts/grid_search.py) showed naive val-excess selection overfits under
regime shift (greeks-only), whereas the pre-registered tail-weighting hypothesis
(H2: higher CVaR weight + alpha) generalises. This script confirms that winner with
multi-seed stability + paired-bootstrap / Stouffer significance vs delta-vega on the
held-out test split of each universe, and contrasts it with the learned hedgers.

    python scripts/confirm_winner.py
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pl

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ivsh.evaluation.backtest import run_baseline, run_policy
from ivsh.evaluation.stats import paired_bootstrap_diff, stouffer_combine, wilcoxon_pnl
from ivsh.training.objective import cvar_from_pnl
from ivsh.training.train import TrainConfig, fit_prototype, make_standardizer
from ivsh.utils.splits import chronological_split, subset

ROOT = _pl.Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SEEDS = [7, 13, 23, 42, 2025]

# The val-selected winner within the pre-registered tail-weighting hypothesis (H2):
# baseline prototype with a stronger CVaR objective.
WINNER = dict(n_prototypes=8, l2=1e-3, action_scale=1.5, anchor=True,
              cvar_weight=3.0, cvar_alpha=0.975, residual_l2=0.0)


def _load(u):
    with open(ARTIFACTS / f"bank_{u}.pkl", "rb") as f:
        return pickle.load(f)["bank"]


def confirm(uni, learned):
    bank = _load(uni)
    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    scaler = make_standardizer(trb)
    pnls = []
    for s in SEEDS:
        tc = TrainConfig(max_iter=250, seed=s, **WINNER)
        p, _, _ = fit_prototype(trb, scaler, tc, val_bank=vlb)
        pnls.append(run_policy(p, teb, scaler, anchor=True)["pnl"])
    cvars = [cvar_from_pnl(p) for p in pnls]
    proto_pnl = pnls[0]  # seed-7 representative for paired tests
    dv_pnl = run_baseline(teb, "delta_vega")["pnl"]
    bs = paired_bootstrap_diff(proto_pnl, dv_pnl, stat="cvar")
    wl = wilcoxon_pnl(proto_pnl, dv_pnl)
    row = {
        "universe": uni,
        "proto_cvar_mean": float(np.mean(cvars)), "proto_cvar_std": float(np.std(cvars)),
        "delta_vega_cvar": cvar_from_pnl(dv_pnl),
        "dcvar_vs_dv": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
        "p_boot_vs_dv": bs["p_two_sided"], "p_wilcoxon_vs_dv": wl["pvalue"],
        "mlp_cvar": learned.get((uni, "blackbox"), np.nan),
        "ppo_cvar": learned.get((uni, "ppo"), np.nan),
        "sac_cvar": learned.get((uni, "sac"), np.nan),
    }
    return row, bs


def main() -> None:
    # learned-hedger test CVaRs from the earlier multi-universe run
    comp = pd.read_csv(ROOT / "reports_real" / "tables" / "multiverse_comparison.csv")
    learned = {(r.universe, r.method): r.cvar_95 for r in comp.itertuples()}

    rows, bss = [], []
    for u in ["spy", "qqq"]:
        r, bs = confirm(u, learned)
        rows.append(r); bss.append(bs)
        print(f"[{u}] prototype(tail-weighted) CVaR95 = {r['proto_cvar_mean']:.3f}"
              f"±{r['proto_cvar_std']:.3f} | delta_vega {r['delta_vega_cvar']:.3f} | "
              f"dCVaR {r['dcvar_vs_dv']:+.3f} (p={r['p_boot_vs_dv']:.3f}) | "
              f"MLP {r['mlp_cvar']:.2f} PPO {r['ppo_cvar']:.2f} SAC {r['sac_cvar']:.2f}")
    comb = stouffer_combine(bss)
    print(f"\nStouffer-combined prototype - delta_vega across universes: "
          f"mean_d={comb['mean_diff']:+.3f}, combined p={comb['p_two_sided']:.4f}")
    df = pd.DataFrame(rows)
    out = ROOT / "reports_real" / "tables" / "winner_confirmation.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    # verdict
    tie_or_beat = all(r["dcvar_vs_dv"] <= 0.05 * r["delta_vega_cvar"] + 1e-9 for r in rows)
    dom_deeprl = all(r["proto_cvar_mean"] < min(r["ppo_cvar"], r["sac_cvar"]) for r in rows)
    print(f"\nBAR: tie-or-beat delta_vega on both universes: {tie_or_beat}")
    print(f"BAR: dominate PPO & SAC on both universes:     {dom_deeprl}")


if __name__ == "__main__":
    main()
