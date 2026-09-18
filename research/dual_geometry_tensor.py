"""Cortex v1.3-R dual-geometry tensor bridge reference."""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Sequence

import numpy as np


def relation_surprisal(
    global_counts: Sequence[float],
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    counts = np.asarray(global_counts, dtype=float)
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("global_counts must be a non-empty vector")
    if np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite and non-negative")
    probs = (counts + alpha) / (counts.sum() + alpha * len(counts))
    return -np.log(probs)


def rarity_weights(
    surprisal: Sequence[float],
    eta: float,
    *,
    bandwidth: float = 0.35,
) -> np.ndarray:
    s = np.asarray(surprisal, dtype=float)
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and positive")
    if not np.isfinite(eta):
        raise ValueError("eta must be finite")
    return np.exp(-0.5 * ((s - eta) / bandwidth) ** 2)


def interaction_features(
    rates: Sequence[float],
    surprisal: Sequence[float],
    eta: float,
    *,
    bandwidth: float = 0.35,
) -> np.ndarray:
    p = np.asarray(rates, dtype=float)
    if p.ndim != 1 or np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("rates must be a finite non-negative vector")
    w = rarity_weights(surprisal, eta, bandwidth=bandwidth)
    if p.shape != w.shape:
        raise ValueError("rates and surprisal must have matching shapes")
    return np.sqrt(w * p)


def interaction_distance(
    a_rates: Sequence[float],
    b_rates: Sequence[float],
    surprisal: Sequence[float],
    eta: float,
    *,
    bandwidth: float = 0.35,
) -> float:
    a = interaction_features(a_rates, surprisal, eta, bandwidth=bandwidth)
    b = interaction_features(b_rates, surprisal, eta, bandwidth=bandwidth)
    return float(np.linalg.norm(a - b) / sqrt(2.0))


@dataclass(frozen=True)
class TensorBridge:
    tensor: np.ndarray
    relation_count: int
    semantic_dim: int

    def predict(self, source: Sequence[float], target: Sequence[float]) -> np.ndarray:
        s = np.asarray(source, dtype=float)
        t = np.asarray(target, dtype=float)
        if s.shape != (self.semantic_dim,) or t.shape != (self.semantic_dim,):
            raise ValueError("semantic vector dimension mismatch")
        raw = np.einsum("a,abr,b->r", s, self.tensor, t)
        return np.maximum(raw, 0.0)


def fit_tensor_bridge(
    semantic_features: Sequence[Sequence[float]],
    samples: Sequence[tuple[int, int, Sequence[float]]],
    *,
    ridge: float = 1.0e-4,
) -> TensorBridge:
    semantics = np.asarray(semantic_features, dtype=float)
    if semantics.ndim != 2 or not np.all(np.isfinite(semantics)):
        raise ValueError("semantic_features must be a finite 2D matrix")
    if not samples:
        raise ValueError("samples cannot be empty")

    d = semantics.shape[1]
    relation_count = len(samples[0][2])
    x_rows = []
    y_rows = []
    for source, target, rates in samples:
        y = np.asarray(rates, dtype=float)
        if y.shape != (relation_count,):
            raise ValueError("inconsistent relation target dimension")
        if np.any(y < 0) or not np.all(np.isfinite(y)):
            raise ValueError("relation targets must be finite and non-negative")
        x_rows.append(np.kron(semantics[source], semantics[target]))
        y_rows.append(y)

    x = np.asarray(x_rows)
    y = np.asarray(y_rows)
    regularizer = ridge * np.eye(x.shape[1])
    coefficients = np.linalg.solve(x.T @ x + regularizer, x.T @ y)

    tensor = np.zeros((d, d, relation_count), dtype=float)
    for relation in range(relation_count):
        tensor[:, :, relation] = coefficients[:, relation].reshape(d, d)
    return TensorBridge(tensor, relation_count, d)


def resolved_event_scores(
    bridge: TensorBridge,
    source: Sequence[float],
    target: Sequence[float],
    surprisal: Sequence[float],
    eta: float,
    *,
    direct_rates: Sequence[float] | None = None,
    direct_confidence: Sequence[float] | None = None,
    bandwidth: float = 0.35,
) -> np.ndarray:
    bridge_rates = bridge.predict(source, target)
    if direct_rates is None:
        fused = bridge_rates
    else:
        direct = np.asarray(direct_rates, dtype=float)
        confidence = np.asarray(direct_confidence, dtype=float)
        if direct.shape != bridge_rates.shape or confidence.shape != bridge_rates.shape:
            raise ValueError("direct evidence dimension mismatch")
        if np.any(confidence < 0) or np.any(confidence > 1):
            raise ValueError("confidence must be in [0,1]")
        fused = confidence * direct + (1.0 - confidence) * bridge_rates

    return rarity_weights(
        surprisal,
        eta,
        bandwidth=bandwidth,
    ) * fused


def toy_fixture():
    # Semantic coordinates: key-like, lock-like, tool/fastener.
    names = ["key-a", "key-b", "lock-a", "lock-b", "hammer", "nail"]
    semantics = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.2, 0.98],
        ],
        dtype=float,
    )

    # Relation types: near (common), used-with (medium), unlock (rare).
    global_counts = np.asarray([1000.0, 180.0, 20.0])
    surprisal = relation_surprisal(global_counts)

    # Hold out key-b -> lock-b entirely.
    samples = [
        (0, 2, [0.55, 0.18, 0.82]),
        (0, 3, [0.52, 0.16, 0.78]),
        (1, 2, [0.53, 0.17, 0.80]),
        (4, 5, [0.50, 0.83, 0.02]),
        (4, 2, [0.60, 0.05, 0.01]),
        (0, 5, [0.58, 0.05, 0.01]),
    ]
    heldout = (1, 3)
    return names, semantics, global_counts, surprisal, samples, heldout
