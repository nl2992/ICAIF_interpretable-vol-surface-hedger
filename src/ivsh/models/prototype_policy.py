from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PrototypePolicy:
    """Similarity-weighted prototype action head."""

    prototypes: np.ndarray
    actions: np.ndarray
    temperature: float = 0.35

    def __post_init__(self) -> None:
        if self.prototypes.ndim != 2:
            raise ValueError("prototypes must be [n_prototypes, n_features]")
        if self.actions.ndim != 2:
            raise ValueError("actions must be [n_prototypes, action_dim]")
        if self.prototypes.shape[0] != self.actions.shape[0]:
            raise ValueError("prototype and action counts must match")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def weights(self, states: np.ndarray) -> np.ndarray:
        distances = ((states[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(axis=-1)
        logits = -distances / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self.weights(states) @ self.actions


class ProtoSurfaceHedger:
    """Trainable, interpretable prototype hedging policy.

    Prototypes live in the standardised feature space and are held fixed (they
    are discovered by clustering training states). The learnable parameters are a
    bounded hedge action per prototype and a global similarity temperature. The
    hedge action for any state is the similarity-weighted average of prototype
    actions, which is exactly the quantity surfaced for interpretability.
    """

    def __init__(
        self,
        prototypes: np.ndarray,
        action_dim: int = 2,
        action_scale: float = 4.0,
        init_log_temp: float = 0.0,
        seed: int = 7,
    ) -> None:
        self.prototypes = np.asarray(prototypes, dtype=float)
        self.k, self.n_features = self.prototypes.shape
        self.action_dim = action_dim
        self.action_scale = action_scale
        rng = np.random.default_rng(seed)
        self.raw_actions = 0.01 * rng.standard_normal((self.k, action_dim))
        self.log_temp = float(init_log_temp)

    # --- learnable parameter plumbing ---------------------------------------
    @property
    def n_params(self) -> int:
        return self.k * self.action_dim + 1

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([self.raw_actions.ravel(), [self.log_temp]])

    def set_flat_params(self, theta: np.ndarray) -> None:
        self.raw_actions = theta[:-1].reshape(self.k, self.action_dim).copy()
        self.log_temp = float(theta[-1])

    # --- forward pass --------------------------------------------------------
    @property
    def actions(self) -> np.ndarray:
        """Bounded, interpretable hedge action per prototype."""
        return np.tanh(self.raw_actions) * self.action_scale

    def weights(self, states_std: np.ndarray) -> np.ndarray:
        temp = np.exp(self.log_temp) + 1e-6
        dist = ((states_std[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(axis=-1)
        logits = -dist / temp
        logits -= logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return ex / ex.sum(axis=1, keepdims=True)

    def predict_holdings(self, states_std: np.ndarray) -> np.ndarray:
        return self.weights(states_std) @ self.actions

    # --- forward/backward for analytic training -----------------------------
    def forward(self, x: np.ndarray):
        temp = np.exp(self.log_temp) + 1e-6
        dist = ((x[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(axis=-1)
        logits = -dist / temp
        logits -= logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        w = ex / ex.sum(axis=1, keepdims=True)
        actions = np.tanh(self.raw_actions) * self.action_scale
        holdings = w @ actions
        cache = (w, actions, dist, temp)
        return holdings, cache

    def backward(self, grad_holdings: np.ndarray, cache) -> np.ndarray:
        w, actions, dist, temp = cache
        tanh = actions / self.action_scale
        d_actions = w.T @ grad_holdings  # [K, action_dim]
        d_raw = d_actions * self.action_scale * (1.0 - tanh**2)
        dw = grad_holdings @ actions.T  # [N, K]
        # softmax backward
        dlogits = w * (dw - (dw * w).sum(axis=1, keepdims=True))
        # logits = -dist / temp  ->  d logits / d temp = dist / temp**2
        d_temp = float((dlogits * (dist / temp**2)).sum())
        d_log_temp = d_temp * (temp - 1e-6)
        return np.concatenate([d_raw.ravel(), [d_log_temp]])

