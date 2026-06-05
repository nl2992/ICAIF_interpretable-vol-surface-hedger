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

