"""Small PPO robustness grid with position-limited residual actions."""

from __future__ import annotations

import argparse
import itertools
import pathlib as _pl
import pickle
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from ivsh.models.deep_rl import RLConfig, evaluate_sb3, train_sb3
from ivsh.training.objective import cvar_from_pnl
from ivsh.training.train import make_standardizer
from ivsh.utils.splits import chronological_split, subset

ROOT = _pl.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="spy")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-configs", type=int, default=9)
    ap.add_argument("--timesteps", type=int, default=60_000)
    ap.add_argument("--reports-dir", default="reports_real")
    args = ap.parse_args()

    with open(ROOT / "artifacts" / f"bank_{args.universe}.pkl", "rb") as f:
        bank = pickle.load(f)["bank"]
    sp = chronological_split(bank)
    trb, vlb, teb = subset(bank, sp.train), subset(bank, sp.val), subset(bank, sp.test)
    scaler = make_standardizer(trb)

    lrs = [1e-4, 3e-4, 1e-3]
    scales = [0.75, 1.5, 2.5]
    configs = list(itertools.product(lrs, scales))[: args.n_configs]
    rows = []
    for seed in range(args.seeds):
        for lr, scale in configs:
            cfg = RLConfig(algo="ppo", total_timesteps=args.timesteps, learning_rate=lr,
                           action_scale=scale, seed=7 + seed, device="auto")
            model = train_sb3(trb, scaler, cfg)
            val = evaluate_sb3(model, vlb, scaler, action_scale=scale)["pnl"]
            test = evaluate_sb3(model, teb, scaler, action_scale=scale)["pnl"]
            row = {
                "universe": args.universe,
                "seed": 7 + seed,
                "learning_rate": lr,
                "action_scale": scale,
                "val_cvar95": cvar_from_pnl(val),
                "test_cvar95": cvar_from_pnl(test),
                "test_mean": float(test.mean()),
            }
            rows.append(row)
            print(f"seed={row['seed']} lr={lr:g} scale={scale:g} val={row['val_cvar95']:.3f} test={row['test_cvar95']:.3f}")

    out = ROOT / args.reports_dir / "tables" / f"tuned_ppo_{args.universe}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
