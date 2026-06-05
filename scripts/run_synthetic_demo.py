from __future__ import annotations

import numpy as np

from ivsh.backtest.simple import hedge_pnl
from ivsh.data.synthetic import make_synthetic_surfaces
from ivsh.evaluation.risk import cvar
from ivsh.features.surface import flatten_surfaces, surface_tensor
from ivsh.models.prototype_policy import PrototypePolicy


def main() -> None:
    frame = make_synthetic_surfaces()
    states = flatten_surfaces(surface_tensor(frame))
    prototypes = states[[10, 35, 50, 75, 95, 110]]
    actions = np.array(
        [
            [0.10, -0.05],
            [0.05, -0.02],
            [-0.20, 0.15],
            [0.00, 0.00],
            [0.08, -0.04],
            [0.03, -0.01],
        ]
    )
    policy = PrototypePolicy(prototypes=prototypes, actions=actions)
    predicted = policy.predict(states)
    risk_moves = np.column_stack([np.diff(states[:, 0], prepend=states[0, 0]), np.diff(states[:, -1], prepend=states[0, -1])])
    pnl = hedge_pnl(predicted, risk_moves)
    print({"steps": len(pnl), "mean_pnl": round(float(pnl.mean()), 8), "cvar_95": round(cvar(pnl), 8)})


if __name__ == "__main__":
    main()

