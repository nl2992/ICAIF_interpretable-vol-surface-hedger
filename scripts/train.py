"""Stage 2 — train the prototype and black-box hedgers on the prepared dataset.

    python scripts/train.py --config configs/experiment.yaml

Reads artifacts/dataset.pkl (run build_dataset.py first) and writes the trained
models to artifacts/models.pkl.
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))


import argparse
import pickle
from pathlib import Path

from ivsh.pipeline import load_config, train_models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", default="artifacts/dataset.pkl")
    ap.add_argument("--out", default="artifacts/models.pkl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.dataset, "rb") as fh:
        data = pickle.load(fh)["data"]
    models = train_models(cfg, data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"models": models}, fh)
    print("prototype:", models["proto_hist"])
    print("blackbox :", models["bb_hist"])
    print(f"models -> {args.out}")


if __name__ == "__main__":
    main()
