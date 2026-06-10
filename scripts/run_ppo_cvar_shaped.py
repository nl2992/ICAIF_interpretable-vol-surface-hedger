"""Does a CVaR-shaped training reward rescue PPO's tail blow-up?

Reviewer point: the unshaped PPO blow-up (test CVaR95 ~35.7 vs prototype 2.34)
could be a straw man, since PPO optimises mean reward. This trains PPO with a
downside-amplified per-step reward (reward -> reward + kappa*min(reward,0)) and
reports the TRUE-P&L test CVaR95 (computed independently of the training reward
via bank.episode_pnl) across seeds, for kappa in {0, 10, 50}. The prototype
hedger remains the reference at 2.34.

Output: reports_real/tables/ppo_cvar_shaped_<universe>.csv
"""

from __future__ import annotations

import argparse
import pathlib as _pl
import pickle
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from ivsh.models.deep_rl import RLConfig, evaluate_sb3, train_sb3
from ivsh.training.objective import cvar_from_pnl
from ivsh.training.train import make_standardizer
from ivsh.utils.splits import chronological_split, subset

ROOT = _pl.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="spy")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--timesteps", type=int, default=60_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.0, 10.0, 50.0])
    ap.add_argument("--reports-dir", default="reports_real")
    args = ap.parse_args()

    with open(ROOT / "artifacts" / f"bank_{args.universe}.pkl", "rb") as f:
        bank = pickle.load(f)["bank"]
    sp = chronological_split(bank)
    trb, teb = subset(bank, sp.train), subset(bank, sp.test)
    scaler = make_standardizer(trb)

    rows = []
    for kappa in args.kappas:
        for seed in range(args.seeds):
            cfg = RLConfig(
                algo="ppo", total_timesteps=args.timesteps, learning_rate=args.lr,
                action_scale=args.scale, seed=7 + seed, device="auto",
                downside_kappa=kappa,
            )
            model = train_sb3(trb, scaler, cfg)
            test = evaluate_sb3(model, teb, scaler, action_scale=args.scale)["pnl"]
            row = {
                "universe": args.universe,
                "downside_kappa": kappa,
                "seed": 7 + seed,
                "test_cvar95": cvar_from_pnl(test),
                "test_mean": float(np.asarray(test).mean()),
            }
            rows.append(row)
            print(f"kappa={kappa:g} seed={row['seed']} test_cvar95={row['test_cvar95']:.3f} mean={row['test_mean']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / args.reports_dir / "tables" / f"ppo_cvar_shaped_{args.universe}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summ = (df.groupby("downside_kappa")["test_cvar95"]
              .agg(["mean", "std", "min", "max"]).round(3))
    print("\n=== test CVaR95 by kappa (lower is better; prototype ref = 2.34) ===")
    print(summ.to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
