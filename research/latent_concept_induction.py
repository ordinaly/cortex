"""Latent concept induction by predictive compression.

The learner exhaustively evaluates the frozen set-partition grammar for six
entities and retains every minimum-description partition within a numerical
tie tolerance. Hidden generator labels are never consumed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Sequence

import numpy as np


def canonical_partitions(size: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate set partitions as restricted-growth strings."""
    if size < 1:
        raise ValueError("size must be positive")

    labels = [0] * size
    result: list[tuple[int, ...]] = []

    def visit(index: int, maximum: int) -> None:
        if index == size:
            result.append(tuple(labels))
            return
        for value in range(maximum + 2):
            labels[index] = value
            visit(index + 1, max(maximum, value))

    labels[0] = 0
    visit(1, 0)
    return tuple(result)


def groups_in(partition: Sequence[int]) -> int:
    if not partition:
        raise ValueError("partition must not be empty")
    return max(partition) + 1


def partition_score(
    counts: np.ndarray,
    partition: Sequence[int],
    *,
    alpha: float = 0.5,
) -> float:
    """Frozen BIC-style predictive-compression score."""
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(partition):
        raise ValueError("counts/partition shape mismatch")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("counts must be finite and non-negative")
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")

    outcomes = values.shape[1]
    group_count = groups_in(partition)
    labels = np.asarray(partition, dtype=int)
    nll = 0.0

    for group in range(group_count):
        mask = labels == group
        pooled = values[mask].sum(axis=0)
        probabilities = (
            pooled + alpha
        ) / (
            float(pooled.sum()) + alpha * outcomes
        )
        nll -= float(
            np.sum(
                values[mask]
                * np.log(probabilities)
            )
        )

    total = max(float(values.sum()), 2.0)
    complexity = (
        0.5
        * group_count
        * (outcomes - 1)
        * log(total)
    )
    return nll + complexity


def probabilities_for_partition(
    counts: np.ndarray,
    partition: Sequence[int],
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    outcomes = values.shape[1]
    labels = np.asarray(partition, dtype=int)
    result = np.zeros_like(values, dtype=float)

    for group in range(groups_in(partition)):
        mask = labels == group
        pooled = values[mask].sum(axis=0)
        probability = (
            pooled + alpha
        ) / (
            float(pooled.sum()) + alpha * outcomes
        )
        result[mask] = probability

    return result


@dataclass(frozen=True)
class ConceptPrediction:
    probabilities: tuple[float, ...] | None
    resolved: bool
    version_space: int


@dataclass(frozen=True)
class ConceptFit:
    best_score: float
    partitions: tuple[tuple[int, ...], ...]
    candidate_count: int

    @property
    def resolved(self) -> bool:
        return len(self.partitions) == 1

    @property
    def concept_count(self) -> int | None:
        if not self.resolved:
            return None
        return groups_in(self.partitions[0])


class CortexLatentConceptInducer:
    def __init__(
        self,
        counts: np.ndarray,
        *,
        alpha: float = 0.5,
        tie_tolerance: float = 1.0e-9,
    ) -> None:
        values = np.asarray(counts, dtype=float)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("counts must be a non-empty matrix")
        if np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError("counts must be finite and non-negative")
        if tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative")

        self.counts = values.copy()
        self.alpha = float(alpha)
        self.tie_tolerance = float(tie_tolerance)
        self.candidates = canonical_partitions(values.shape[0])
        scored = [
            (
                partition_score(
                    self.counts,
                    partition,
                    alpha=self.alpha,
                ),
                partition,
            )
            for partition in self.candidates
        ]
        best = min(score for score, _partition in scored)
        best_partitions = tuple(
            partition
            for score, partition in scored
            if abs(score - best) <= self.tie_tolerance
        )
        self.fit = ConceptFit(
            best_score=best,
            partitions=best_partitions,
            candidate_count=len(self.candidates),
        )

    def co_membership(
        self,
        left: int,
        right: int,
    ) -> float:
        if not (
            0 <= left < len(self.counts)
            and 0 <= right < len(self.counts)
        ):
            raise ValueError("entity index out of range")
        return sum(
            int(partition[left] == partition[right])
            for partition in self.fit.partitions
        ) / len(self.fit.partitions)

    def predict_entity(
        self,
        entity: int,
    ) -> ConceptPrediction:
        if not 0 <= entity < len(self.counts):
            raise ValueError("entity index out of range")

        predictions = [
            probabilities_for_partition(
                self.counts,
                partition,
                alpha=self.alpha,
            )[entity]
            for partition in self.fit.partitions
        ]
        first = predictions[0]
        if all(
            np.allclose(
                first,
                candidate,
                atol=1.0e-12,
                rtol=1.0e-12,
            )
            for candidate in predictions[1:]
        ):
            return ConceptPrediction(
                probabilities=tuple(
                    float(value)
                    for value in first
                ),
                resolved=True,
                version_space=len(predictions),
            )
        return ConceptPrediction(
            probabilities=None,
            resolved=False,
            version_space=len(predictions),
        )

    def representative_probabilities(self) -> np.ndarray:
        if not self.fit.resolved:
            raise ValueError(
                "latent partition is unresolved"
            )
        return probabilities_for_partition(
            self.counts,
            self.fit.partitions[0],
            alpha=self.alpha,
        )


def flat_probabilities(
    counts: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    outcomes = values.shape[1]
    totals = values.sum(axis=1, keepdims=True)
    return (
        values + alpha
    ) / (
        totals + alpha * outcomes
    )


def global_probabilities(
    counts: np.ndarray,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    values = np.asarray(counts, dtype=float)
    outcomes = values.shape[1]
    pooled = values.sum(axis=0)
    probability = (
        pooled + alpha
    ) / (
        float(pooled.sum()) + alpha * outcomes
    )
    return np.repeat(
        probability[None, :],
        len(values),
        axis=0,
    )


def heldout_nll(
    probabilities: np.ndarray,
    heldout_counts: np.ndarray,
) -> float:
    p = np.asarray(probabilities, dtype=float)
    heldout = np.asarray(heldout_counts, dtype=float)
    if p.shape != heldout.shape:
        raise ValueError("probability/heldout shape mismatch")
    total = float(heldout.sum())
    if total <= 0:
        raise ValueError("heldout counts must have positive mass")
    return -float(
        np.sum(
            heldout * np.log(
                np.maximum(p, 1.0e-15)
            )
        )
    ) / total


def pairwise_partition_f1(
    predicted: Sequence[int],
    truth: Sequence[int],
) -> float:
    if len(predicted) != len(truth):
        raise ValueError("partition sizes differ")
    tp = fp = fn = 0
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            same_truth = truth[left] == truth[right]
            same_predicted = (
                predicted[left] == predicted[right]
            )
            if same_truth and same_predicted:
                tp += 1
            elif not same_truth and same_predicted:
                fp += 1
            elif same_truth and not same_predicted:
                fn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall <= 1.0e-15:
        return 1.0 if tp == fp == fn == 0 else 0.0
    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )
