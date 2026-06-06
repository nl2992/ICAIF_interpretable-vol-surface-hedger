"""Black-box deep-hedging baseline: a small numpy MLP policy.

A one-hidden-layer tanh network mapping standardised state features to bounded
hedge holdings. It shares the prototype model's inputs, action space, cost model
and CVaR objective, so the only difference is interpretability — it is the
"competitive black box" the prototype hedger is measured against.
"""

from __future__ import annotations

import numpy as np


class MLPHedger:
    def __init__(
        self,
        n_features: int,
        hidden: int = 16,
        action_dim: int = 2,
        action_scale: float = 4.0,
        seed: int = 7,
    ) -> None:
        self.n_features = n_features
        self.hidden = hidden
        self.action_dim = action_dim
        self.action_scale = action_scale
        rng = np.random.default_rng(seed)
        # He-ish small init keeps initial actions near zero.
        self.w1 = 0.1 * rng.standard_normal((n_features, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = 0.1 * rng.standard_normal((hidden, action_dim))
        self.b2 = np.zeros(action_dim)

    @property
    def n_params(self) -> int:
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate(
            [self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()]
        )

    def set_flat_params(self, theta: np.ndarray) -> None:
        i = 0
        n = self.w1.size
        self.w1 = theta[i : i + n].reshape(self.w1.shape).copy()
        i += n
        n = self.b1.size
        self.b1 = theta[i : i + n].copy()
        i += n
        n = self.w2.size
        self.w2 = theta[i : i + n].reshape(self.w2.shape).copy()
        i += n
        self.b2 = theta[i:].copy()

    def predict_holdings(self, states_std: np.ndarray) -> np.ndarray:
        h = np.tanh(states_std @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return np.tanh(out) * self.action_scale

    # --- forward/backward for analytic training -----------------------------
    def forward(self, x: np.ndarray):
        a1 = x @ self.w1 + self.b1
        h = np.tanh(a1)
        a2 = h @ self.w2 + self.b2
        out = np.tanh(a2)
        holdings = out * self.action_scale
        cache = (x, h, a2)
        return holdings, cache

    def backward(self, grad_holdings: np.ndarray, cache) -> np.ndarray:
        x, h, a2 = cache
        da2 = grad_holdings * self.action_scale * (1.0 - np.tanh(a2) ** 2)
        d_w2 = h.T @ da2
        d_b2 = da2.sum(axis=0)
        dh = da2 @ self.w2.T
        da1 = dh * (1.0 - h**2)
        d_w1 = x.T @ da1
        d_b1 = da1.sum(axis=0)
        return np.concatenate([d_w1.ravel(), d_b1.ravel(), d_w2.ravel(), d_b2.ravel()])
