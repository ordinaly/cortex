"""Cortex v1.6-R fuzzy predictive field reference."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Sequence

import numpy as np

from dual_geometry_tensor import (
    TensorBridge,
    fit_tensor_bridge,
    rarity_weights,
    relation_surprisal,
)


def _normalize(vector: Sequence[float]) -> np.ndarray:
    x = np.asarray(vector, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("vector must be finite and non-empty")
    norm = float(np.linalg.norm(x))
    if norm <= 1.0e-15:
        raise ValueError("zero vector")
    return x / norm


def responsibilities(possibility: Sequence[float]) -> np.ndarray:
    values = np.asarray(possibility, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("possibility must be a non-empty vector")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("possibility must be finite and non-negative")
    total = float(values.sum())
    if total <= 1.0e-15:
        return np.full(len(values), 1.0 / len(values))
    return values / total


def perceptual_possibility(
    observation: Sequence[float],
    anchors: Sequence[Sequence[float]],
    *,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    x = _normalize(observation)
    a = np.asarray([_normalize(row) for row in anchors], dtype=float)
    distances = np.maximum(0.0, 1.0 - a @ x)
    return np.exp(-distances / temperature)


def semantic_features(
    counts: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    if counts.ndim != 2 or np.any(counts < 0):
        raise ValueError("counts must be a non-negative matrix")
    probs = (counts + alpha) / (
        counts.sum(axis=1, keepdims=True) + alpha * counts.shape[1]
    )
    return np.sqrt(probs)


def predictive_distance_matrix(features: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - features[None, :, :]
    return np.linalg.norm(diff, axis=2) / np.sqrt(2.0)


def predictive_kernel(
    features: np.ndarray,
    *,
    resolution: float,
) -> np.ndarray:
    if resolution <= 0 or not np.isfinite(resolution):
        raise ValueError("resolution must be finite and positive")
    distances = predictive_distance_matrix(features)
    return np.exp(-0.5 * (distances / resolution) ** 2)


def effective_rank(kernel: np.ndarray) -> float:
    symmetric = 0.5 * (kernel + kernel.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 1.0e-15:
        return 0.0
    p = eigenvalues / total
    p = p[p > 1.0e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


@dataclass
class Prediction:
    scores: np.ndarray
    source_field: np.ndarray
    target_field: np.ndarray


class FuzzyPredictiveField:
    def __init__(
        self,
        anchors: Sequence[Sequence[float]],
        *,
        outcomes: int,
        relations: int,
        perceptual_temperature: float = 0.05,
        semantic_resolution: float = 0.10,
        semantic_alpha: float = 0.5,
        evidence_decay: float = 1.0,
    ) -> None:
        self.anchors = np.asarray(
            [_normalize(anchor) for anchor in anchors],
            dtype=float,
        )
        self.outcomes = outcomes
        self.relations = relations
        self.perceptual_temperature = perceptual_temperature
        self.semantic_resolution = semantic_resolution
        self.semantic_alpha = semantic_alpha
        self.evidence_decay = evidence_decay

        n = len(self.anchors)
        self.outcome_counts = np.zeros((n, outcomes), dtype=float)
        self.pair_relation_counts = np.zeros(
            (n, n, relations),
            dtype=float,
        )
        self.global_relation_counts = np.zeros(relations, dtype=float)

    def possibility(self, observation: Sequence[float]) -> np.ndarray:
        return perceptual_possibility(
            observation,
            self.anchors,
            temperature=self.perceptual_temperature,
        )

    def perceptual_weights(self, observation: Sequence[float]) -> np.ndarray:
        return responsibilities(self.possibility(observation))

    def features(self) -> np.ndarray:
        return semantic_features(
            self.outcome_counts,
            alpha=self.semantic_alpha,
        )

    def kernel(self, *, resolution: float | None = None) -> np.ndarray:
        return predictive_kernel(
            self.features(),
            resolution=(
                self.semantic_resolution
                if resolution is None
                else resolution
            ),
        )

    def field(
        self,
        observation: Sequence[float],
        *,
        resolution: float | None = None,
    ) -> np.ndarray:
        weights = self.perceptual_weights(observation)
        activation = weights @ self.kernel(resolution=resolution)
        return responsibilities(activation)

    def observe_outcome(
        self,
        observation: Sequence[float],
        outcome: int,
    ) -> None:
        if not 0 <= outcome < self.outcomes:
            raise ValueError("invalid outcome")
        if self.evidence_decay < 1.0:
            self.outcome_counts *= self.evidence_decay
        weights = self.perceptual_weights(observation)
        self.outcome_counts[:, outcome] += weights

    def observe_relation(
        self,
        source_observation: Sequence[float],
        target_observation: Sequence[float],
        relation: int,
    ) -> None:
        if not 0 <= relation < self.relations:
            raise ValueError("invalid relation")
        if self.evidence_decay < 1.0:
            self.pair_relation_counts *= self.evidence_decay
            self.global_relation_counts *= self.evidence_decay
        source = self.perceptual_weights(source_observation)
        target = self.perceptual_weights(target_observation)
        self.pair_relation_counts[:, :, relation] += np.outer(
            source,
            target,
        )
        self.global_relation_counts[relation] += 1.0

    def bridge(self, *, minimum_mass: float = 1.0e-4) -> TensorBridge:
        features = self.features()
        samples = []
        totals = self.pair_relation_counts.sum(axis=2)
        for i in range(len(self.anchors)):
            for j in range(len(self.anchors)):
                total = float(totals[i, j])
                if total <= minimum_mass:
                    continue
                rates = self.pair_relation_counts[i, j] / total
                samples.append((i, j, rates.tolist()))
        if not samples:
            raise ValueError("insufficient relational evidence")
        return fit_tensor_bridge(features, samples, ridge=1.0e-3)

    def predict(
        self,
        source_observation: Sequence[float],
        target_observation: Sequence[float],
        *,
        eta: float | None = None,
        interaction_bandwidth: float = 0.45,
        resolution: float | None = None,
    ) -> Prediction:
        features = self.features()
        source_field = self.field(
            source_observation,
            resolution=resolution,
        )
        target_field = self.field(
            target_observation,
            resolution=resolution,
        )
        source_semantic = source_field @ features
        target_semantic = target_field @ features

        bridge = self.bridge()
        rates = bridge.predict(source_semantic, target_semantic)

        surprisal = relation_surprisal(
            self.global_relation_counts + 1.0
        )
        if eta is None:
            eta = float(np.mean(surprisal[2:]))
        scores = rarity_weights(
            surprisal,
            eta,
            bandwidth=interaction_bandwidth,
        ) * rates
        return Prediction(scores, source_field, target_field)

    def predictive_complexity(
        self,
        *,
        resolution: float | None = None,
    ) -> float:
        return effective_rank(self.kernel(resolution=resolution))
