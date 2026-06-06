"""Stage 1 — simulate the market and build train/val/test episode banks.

    python scripts/build_dataset.py --config configs/experiment.yaml

Writes the dataset artifact to artifacts/dataset.pkl for the downstream stages.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))


import argparse
import pickle
from pathlib import Path

from ivsh.pipeline import build_data, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="artifacts/dataset.pkl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = build_data(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"cfg": cfg, "data": data}, fh)
    print(
        f"built {data['train'].n_episodes} train / {data['val'].n_episodes} val / "
        f"{data['test'].n_episodes} test episodes -> {args.out}"
    )


if __name__ == "__main__":
    main()
