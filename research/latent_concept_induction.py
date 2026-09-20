"""Latent concept induction by predictive compression.

Research-only Cortex Reasoning Map prototype.

The learner receives entity/outcome evidence but no concept labels. It evaluates
the complete finite set-partition grammar, retains exact minimum-description
ties as a version space, and treats only blocks shared by every optimum as
resolved latent concepts.

A small lifecycle wrapper gives resolved concept blocks persistent IDs across
successive evidence snapshots and records create/split/merge/retract events.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Iterable, Sequence

import numpy as np


Partition = tuple[int, ...]
ConceptMembers = frozenset[int]


def canonical_partitions(size: int) -> tuple[Partition, ...]:
    """Enumerate set partitions as restricted-growth strings."""
    if size < 1:
        raise ValueError("size must be positive")

    labels = [0] * size
    result: list[Partition] = []

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


def partition_blocks(
    partition: Sequence[int],
) -> tuple[ConceptMembers, ...]:
    grouped: dict[int, set[int]] = {}
    for entity, group in enumerate(partition):
        grouped.setdefault(int(group), set()).add(entity)
    return tuple(
        frozenset(grouped[group])
        for group in sorted(grouped)
    )


def partition_score(
    counts: np.ndarray,
    partition: Sequence[int],
    *,
    alpha: float = 0.5,
) -> float:
    """BIC-style predictive-description length for one partition."""
    values = np.asarray(counts, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(partition):
        raise ValueError("counts/partition shape mismatch")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("counts must be finite and non-negative")
    if values.shape[1] < 2:
        raise ValueError("at least two outcomes are required")
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")

    outcomes = values.shape[1]
    labels = np.asarray(partition, dtype=int)
    group_count = groups_in(partition)
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
                * np.log(np.maximum(probabilities, 1.0e-15))
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
    if values.ndim != 2 or values.shape[0] != len(partition):
        raise ValueError("counts/partition shape mismatch")
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
            heldout
            * np.log(np.maximum(p, 1.0e-15))
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


@dataclass(frozen=True)
class ConceptPrediction:
    probabilities: tuple[float, ...] | None
    resolved: bool
    version_space: int


@dataclass(frozen=True)
class ConceptBlockRead:
    members: ConceptMembers
    exact_support: float
    resolved: bool


@dataclass(frozen=True)
class ConceptFit:
    best_score: float
    best_partitions: tuple[Partition, ...]
    best_partition: Partition
    candidate_count: int
    effective_partition_count: float

    @property
    def resolved(self) -> bool:
        return len(self.best_partitions) == 1

    @property
    def concept_count(self) -> int | None:
        if not self.resolved:
            return None
        return groups_in(self.best_partition)


class CortexLatentConceptInducer:
    """Infer a finite latent partition from predictive-description economy."""

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

        scored = tuple(
            (
                partition_score(
                    self.counts,
                    partition,
                    alpha=self.alpha,
                ),
                partition,
            )
            for partition in self.candidates
        )
        best_score = min(score for score, _ in scored)
        best_partitions = tuple(
            partition
            for score, partition in scored
            if abs(score - best_score) <= self.tie_tolerance
        )
        best_partition = min(best_partitions)

        relative_weights = np.asarray(
            [
                exp(-(score - best_score))
                for score, _ in scored
            ],
            dtype=float,
        )
        relative_weights /= relative_weights.sum()

        entropy = -float(
            np.sum(
                relative_weights
                * np.log(
                    np.maximum(relative_weights, 1.0e-300)
                )
            )
        )

        self._scored = scored
        self._weights = relative_weights
        self.fit = ConceptFit(
            best_score=best_score,
            best_partitions=best_partitions,
            best_partition=best_partition,
            candidate_count=len(self.candidates),
            effective_partition_count=exp(entropy),
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
            for partition in self.fit.best_partitions
        ) / len(self.fit.best_partitions)

    def block_support(
        self,
        members: Iterable[int],
    ) -> float:
        block = frozenset(int(member) for member in members)
        if not block:
            raise ValueError("concept block must not be empty")

        support = 0.0
        for weight, (_score, partition) in zip(
            self._weights,
            self._scored,
        ):
            if block in partition_blocks(partition):
                support += float(weight)
        return support

    def resolved_blocks(self) -> tuple[ConceptBlockRead, ...]:
        """Return non-singleton blocks common to every exact optimum."""
        common = set(
            block
            for block in partition_blocks(
                self.fit.best_partitions[0]
            )
            if len(block) >= 2
        )
        for partition in self.fit.best_partitions[1:]:
            common &= {
                block
                for block in partition_blocks(partition)
                if len(block) >= 2
            }

        return tuple(
            ConceptBlockRead(
                members=block,
                exact_support=self.block_support(block),
                resolved=True,
            )
            for block in sorted(
                common,
                key=lambda value: tuple(sorted(value)),
            )
        )

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
            for partition in self.fit.best_partitions
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
            self.fit.best_partition,
            alpha=self.alpha,
        )


@dataclass(frozen=True)
class LatentConcept:
    concept_id: str
    members: ConceptMembers


@dataclass(frozen=True)
class ConceptLifecycleEvent:
    epoch: int
    kind: str
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    source_members: tuple[ConceptMembers, ...]
    target_members: tuple[ConceptMembers, ...]


@dataclass(frozen=True)
class ConceptLifecycleRead:
    epoch: int
    fit: ConceptFit
    active_concepts: tuple[LatentConcept, ...]
    events: tuple[ConceptLifecycleEvent, ...]


class CortexLatentConceptLifecycle:
    """Track persistent concept IDs across successive current-evidence fits."""

    def __init__(
        self,
        entity_count: int,
        *,
        alpha: float = 0.5,
        tie_tolerance: float = 1.0e-9,
    ) -> None:
        if entity_count < 1:
            raise ValueError("entity_count must be positive")
        self.entity_count = int(entity_count)
        self.alpha = float(alpha)
        self.tie_tolerance = float(tie_tolerance)
        self.epoch = -1
        self._next_id = 1
        self._active: dict[ConceptMembers, LatentConcept] = {}
        self._events: list[ConceptLifecycleEvent] = []

    @property
    def events(self) -> tuple[ConceptLifecycleEvent, ...]:
        return tuple(self._events)

    def _allocate(self, members: ConceptMembers) -> LatentConcept:
        concept = LatentConcept(
            concept_id=f"Z{self._next_id}",
            members=members,
        )
        self._next_id += 1
        return concept

    def advance(
        self,
        epoch: int,
        counts: np.ndarray,
    ) -> ConceptLifecycleRead:
        if epoch <= self.epoch:
            raise ValueError("epochs must advance strictly")

        values = np.asarray(counts, dtype=float)
        if values.shape[0] != self.entity_count:
            raise ValueError("entity count changed")

        inducer = CortexLatentConceptInducer(
            values,
            alpha=self.alpha,
            tie_tolerance=self.tie_tolerance,
        )
        new_blocks = {
            read.members
            for read in inducer.resolved_blocks()
        }

        old = dict(self._active)
        preserved = set(old) & new_blocks
        removed = set(old) - new_blocks
        added = new_blocks - set(old)

        new_active: dict[ConceptMembers, LatentConcept] = {
            block: old[block]
            for block in preserved
        }
        for block in sorted(
            added,
            key=lambda value: tuple(sorted(value)),
        ):
            new_active[block] = self._allocate(block)

        step_events: list[ConceptLifecycleEvent] = []
        handled_removed: set[ConceptMembers] = set()
        handled_added: set[ConceptMembers] = set()

        # One old concept becoming two or more resolved child concepts.
        for source_block in sorted(
            removed,
            key=lambda value: tuple(sorted(value)),
        ):
            children = [
                block for block in added
                if block < source_block
            ]
            if (
                len(children) >= 2
                and frozenset().union(*children) == source_block
            ):
                event = ConceptLifecycleEvent(
                    epoch=epoch,
                    kind="split",
                    source_ids=(old[source_block].concept_id,),
                    target_ids=tuple(
                        new_active[block].concept_id
                        for block in sorted(
                            children,
                            key=lambda value: tuple(sorted(value)),
                        )
                    ),
                    source_members=(source_block,),
                    target_members=tuple(
                        sorted(
                            children,
                            key=lambda value: tuple(sorted(value)),
                        )
                    ),
                )
                step_events.append(event)
                handled_removed.add(source_block)
                handled_added.update(children)

        # Two or more old concepts becoming one resolved parent concept.
        for target_block in sorted(
            added - handled_added,
            key=lambda value: tuple(sorted(value)),
        ):
            parents = [
                block for block in removed - handled_removed
                if block < target_block
            ]
            if (
                len(parents) >= 2
                and frozenset().union(*parents) == target_block
            ):
                event = ConceptLifecycleEvent(
                    epoch=epoch,
                    kind="merge",
                    source_ids=tuple(
                        old[block].concept_id
                        for block in sorted(
                            parents,
                            key=lambda value: tuple(sorted(value)),
                        )
                    ),
                    target_ids=(
                        new_active[target_block].concept_id,
                    ),
                    source_members=tuple(
                        sorted(
                            parents,
                            key=lambda value: tuple(sorted(value)),
                        )
                    ),
                    target_members=(target_block,),
                )
                step_events.append(event)
                handled_removed.update(parents)
                handled_added.add(target_block)

        # A strict surviving subset is a one-sided refinement/split.
        for source_block in sorted(
            removed - handled_removed,
            key=lambda value: tuple(sorted(value)),
        ):
            children = [
                block for block in added - handled_added
                if block < source_block
            ]
            if len(children) == 1:
                child = children[0]
                step_events.append(
                    ConceptLifecycleEvent(
                        epoch=epoch,
                        kind="split",
                        source_ids=(old[source_block].concept_id,),
                        target_ids=(
                            new_active[child].concept_id,
                        ),
                        source_members=(source_block,),
                        target_members=(child,),
                    )
                )
                handled_removed.add(source_block)
                handled_added.add(child)

        for source_block in sorted(
            removed - handled_removed,
            key=lambda value: tuple(sorted(value)),
        ):
            step_events.append(
                ConceptLifecycleEvent(
                    epoch=epoch,
                    kind="retracted",
                    source_ids=(old[source_block].concept_id,),
                    target_ids=tuple(),
                    source_members=(source_block,),
                    target_members=tuple(),
                )
            )

        for target_block in sorted(
            added - handled_added,
            key=lambda value: tuple(sorted(value)),
        ):
            step_events.append(
                ConceptLifecycleEvent(
                    epoch=epoch,
                    kind="created",
                    source_ids=tuple(),
                    target_ids=(
                        new_active[target_block].concept_id,
                    ),
                    source_members=tuple(),
                    target_members=(target_block,),
                )
            )

        self.epoch = int(epoch)
        self._active = new_active
        self._events.extend(step_events)

        return ConceptLifecycleRead(
            epoch=self.epoch,
            fit=inducer.fit,
            active_concepts=tuple(
                sorted(
                    self._active.values(),
                    key=lambda concept: concept.concept_id,
                )
            ),
            events=tuple(step_events),
        )
