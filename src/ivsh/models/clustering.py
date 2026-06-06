"""Minimal k-means with k-means++ init (no scikit-learn dependency).

Used to discover volatility-surface *prototypes* in the standardised feature
space. Returns both the cluster centres and the index of the nearest real
training observation to each centre (the medoid), which gives every prototype a
concrete, auditable market date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KMeansResult:
    centers: np.ndarray  # [K, F]
    labels: np.ndarray  # [N]
    medoid_idx: np.ndarray  # [K] index into the fitted data of each medoid
    inertia: float


def _pairwise_sq(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=-1)


def kmeans(x: np.ndarray, k: int, seed: int = 7, max_iter: int = 100) -> KMeansResult:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if k >= n:
        raise ValueError("k must be smaller than the number of samples")

    # k-means++ initialisation.
    centers = np.empty((k, x.shape[1]))
    centers[0] = x[rng.integers(n)]
    closest = ((x - centers[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        probs = closest / closest.sum()
        idx = rng.choice(n, p=probs)
        centers[j] = x[idx]
        closest = np.minimum(closest, ((x - centers[j]) ** 2).sum(axis=1))

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist = _pairwise_sq(x, centers)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)

    dist = _pairwise_sq(x, centers)
    labels = dist.argmin(axis=1)
    inertia = float(dist[np.arange(n), labels].sum())
    medoid_idx = dist.argmin(axis=0)  # nearest sample to each centre
    return KMeansResult(centers=centers, labels=labels, medoid_idx=medoid_idx, inertia=inertia)
