"""Exact small-graph prototype for Cortex v1.1-R semantic geometry.

This module intentionally uses a dense matrix exponential. It is a research
reference only, not an online implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class ResolutionResult:
    rho: float
    features: np.ndarray
    distances: np.ndarray


def sqrt_probability_features(
    channels: Sequence[Sequence[float]],
) -> np.ndarray:
    """Map row-wise probability vectors to the Hellinger feature space."""
    q = np.asarray(channels, dtype=float)
    if q.ndim != 2 or q.shape[1] == 0:
        raise ValueError("channels must be a non-empty 2D matrix")
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("probabilities must be finite and non-negative")
    totals = q.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("each probability row must have positive mass")
    q = q / totals[:, None]
    return np.sqrt(q)


def laplacian(adjacency: Sequence[Sequence[float]]) -> np.ndarray:
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("adjacency must be a non-empty square matrix")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("adjacency must be finite and non-negative")
    if not np.allclose(a, a.T, atol=1e-12):
        raise ValueError("research v0.1 requires symmetric semantic affinity")
    if not np.allclose(np.diag(a), 0.0, atol=1e-12):
        raise ValueError("self-edge weights must be zero")
    return np.diag(a.sum(axis=1)) - a


def semantic_features(
    base_features: Sequence[Sequence[float]],
    adjacency: Sequence[Sequence[float]],
    rho: float,
) -> np.ndarray:
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and non-negative")
    phi0 = np.asarray(base_features, dtype=float)
    if phi0.ndim != 2 or not np.all(np.isfinite(phi0)):
        raise ValueError("base_features must be a finite 2D matrix")
    l = laplacian(adjacency)
    if phi0.shape[0] != l.shape[0]:
        raise ValueError("feature rows must match graph nodes")
    return expm(-rho * l) @ phi0


def pairwise_semantic_distances(features: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - features[None, :, :]
    return np.linalg.norm(diff, axis=2) / sqrt(2.0)


def resolution_result(
    base_features: Sequence[Sequence[float]],
    adjacency: Sequence[Sequence[float]],
    rho: float,
) -> ResolutionResult:
    features = semantic_features(base_features, adjacency, rho)
    return ResolutionResult(
        rho=rho,
        features=features,
        distances=pairwise_semantic_distances(features),
    )


def connection_depth(
    base_features: Sequence[Sequence[float]],
    adjacency: Sequence[Sequence[float]],
    i: int,
    j: int,
    *,
    epsilon: float,
    delta: float = 1.0,
    max_resolution: int = 32,
) -> int | None:
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    if delta <= 0 or not np.isfinite(delta):
        raise ValueError("delta must be finite and positive")
    for resolution in range(max_resolution + 1):
        d = resolution_result(
            base_features,
            adjacency,
            resolution * delta,
        ).distances[i, j]
        if d <= epsilon:
            return resolution
    return None


def hammer_nail_fixture():
    names = ["hammer", "nail", "screwdriver", "moon"]
    role_distributions = [
        [0.85, 0.10, 0.05],
        [0.05, 0.90, 0.05],
        [0.75, 0.20, 0.05],
        [0.05, 0.05, 0.90],
    ]
    adjacency = [
        [0.0, 1.00, 0.00, 0.0],
        [1.00, 0.0, 0.08, 0.0],
        [0.00, 0.08, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    return names, sqrt_probability_features(role_distributions), np.asarray(adjacency)


def demo_rows():
    names, phi0, adjacency = hammer_nail_fixture()
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3)]
    rows = []
    for rho in [0.0, 0.25, 0.5, 1.0, 2.0]:
        result = resolution_result(phi0, adjacency, rho)
        row = {"rho": rho}
        for i, j in pairs:
            row[f"{names[i]}-{names[j]}"] = float(result.distances[i, j])
        rows.append(row)
    return rows


if __name__ == "__main__":
    import json
    for row in demo_rows():
        print(json.dumps(row, sort_keys=True))
