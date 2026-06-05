from __future__ import annotations

import numpy as np

from ivsh.data.synthetic import make_synthetic_surfaces
from ivsh.evaluation.risk import cvar
from ivsh.features.surface import flatten_surfaces, surface_tensor
from ivsh.models.prototype_policy import PrototypePolicy


def test_synthetic_surface_tensor_shape() -> None:
    tensor = surface_tensor(make_synthetic_surfaces(n_steps=12))
    assert tensor.shape == (12, 4, 5)


def test_prototype_policy_weights_are_probabilities() -> None:
    states = flatten_surfaces(surface_tensor(make_synthetic_surfaces(n_steps=20)))
    policy = PrototypePolicy(
        prototypes=states[:3],
        actions=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]),
    )
    weights = policy.weights(states)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert policy.predict(states).shape == (20, 2)


def test_cvar_positive_for_losses() -> None:
    assert cvar(np.array([0.02, -0.01, -0.05, 0.01]), alpha=0.75) == 0.05

