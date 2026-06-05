from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SurfaceGrid:
    moneyness: tuple[float, ...]
    tenor_days: tuple[int, ...]


def make_synthetic_surfaces(
    n_steps: int = 120,
    grid: SurfaceGrid | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Create smooth volatility surfaces with regime shifts for smoke tests."""
    if grid is None:
        grid = SurfaceGrid(moneyness=(0.8, 0.9, 1.0, 1.1, 1.2), tenor_days=(7, 30, 90, 180))

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    level = 0.20
    skew = -0.05

    for t in range(n_steps):
        level = 0.98 * level + 0.02 * 0.20 + rng.normal(0.0, 0.006)
        skew = 0.95 * skew + rng.normal(0.0, 0.004)
        shock = 0.08 if 45 <= t <= 55 else 0.0
        for tenor in grid.tenor_days:
            term = 0.015 * np.log1p(tenor / 30)
            for money in grid.moneyness:
                smile = 0.10 * (money - 1.0) ** 2
                iv = max(0.03, level + shock + skew * (money - 1.0) + term + smile)
                rows.append(
                    {
                        "timestamp": t,
                        "moneyness": money,
                        "tenor_days": tenor,
                        "implied_vol": iv,
                    }
                )
    return pd.DataFrame(rows)

