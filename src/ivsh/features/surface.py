from __future__ import annotations

import numpy as np
import pandas as pd


def surface_tensor(frame: pd.DataFrame) -> np.ndarray:
    """Convert long-form surface rows into [time, tenor, moneyness] tensor."""
    required = {"timestamp", "moneyness", "tenor_days", "implied_vol"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    pivot = frame.pivot_table(
        index="timestamp",
        columns=["tenor_days", "moneyness"],
        values="implied_vol",
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError("surface grid contains missing timestamp/tenor/moneyness cells")

    tenors = pivot.columns.get_level_values(0).unique()
    moneyness = pivot.columns.get_level_values(1).unique()
    return pivot.to_numpy().reshape(len(pivot), len(tenors), len(moneyness))


def flatten_surfaces(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 3:
        raise ValueError("expected [time, tenor, moneyness] tensor")
    return tensor.reshape(tensor.shape[0], -1)

