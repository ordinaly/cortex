"""Research prototype for uncertainty-aware Cortex articulation.

The prototype deliberately lives outside the native hot path. It tests three
mechanisms before any production-core migration:

1. provisional identity hypotheses;
2. delayed evidence commitment;
3. retrospective reconciliation into persistent entities.

The matcher operates in cosine distance over normalized embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _normalize(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("embedding must be a non-empty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError("embedding norm must be non-zero")
    return vector / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, 1.0 - float(np.dot(a, b)))


@dataclass(frozen=True)
class IdentityToken:
    provisional: bool
    identity: int


@dataclass
class _Cluster:
    samples: list[np.ndarray]
    last_seen: int

    @property
    def support(self) -> int:
        return len(self.samples)

    def add(self, sample: np.ndarray, time: int, max_samples: int) -> None:
        self.samples.append(sample)
        if len(self.samples) > max_samples:
            del self.samples[: len(self.samples) - max_samples]
        self.last_seen = time

    def prototype(self) -> np.ndarray:
        median = np.median(np.stack(self.samples, axis=0), axis=0)
        return _normalize(median)


class RobustArticulator:
    """Provisional-identity research layer."""

    def __init__(
        self,
        *,
        accept_distance: float = 0.12,
        provisional_distance: float = 0.94,
        reconcile_distance: float = 0.68,
        commit_support: int = 4,
        provisional_ttl: int = 3,
        max_samples: int = 32,
        max_entities: int = 128,
    ) -> None:
        if not (0 <= accept_distance <= provisional_distance <= 2):
            raise ValueError("invalid matching thresholds")
        if not (0 <= reconcile_distance <= 2):
            raise ValueError("invalid reconcile_distance")
        if commit_support < 2:
            raise ValueError("commit_support must be >= 2")
        self.accept_distance = accept_distance
        self.provisional_distance = provisional_distance
        self.reconcile_distance = reconcile_distance
        self.commit_support = commit_support
        self.provisional_ttl = provisional_ttl
        self.max_samples = max_samples
        self.max_entities = max_entities

        self.time = 0
        self.committed: dict[int, _Cluster] = {}
        self.provisional: dict[int, _Cluster] = {}
        self.reconciled: dict[int, int] = {}
        self.next_committed = 0
        self.next_provisional = 0
        self.reconciliations = 0
        self.new_commits = 0

    def _committed_prototypes(self):
        ids = sorted(self.committed)
        return ids, [self.committed[i].prototype() for i in ids]

    def _active_provisionals(self):
        # Keep provisional state bounded. Reconciled tokens remain resolvable
        # through self.reconciled; stale unresolved tokens intentionally resolve
        # to None because their evidence never earned persistent commitment.
        for identity in list(self.provisional):
            if identity in self.reconciled:
                del self.provisional[identity]
                continue
            if (
                self.time - self.provisional[identity].last_seen
                > self.provisional_ttl
            ):
                del self.provisional[identity]

        ids = sorted(self.provisional)
        return ids, [self.provisional[i].prototype() for i in ids]

    def observe_frame(
        self,
        embeddings: Sequence[Sequence[float]],
    ) -> list[IdentityToken]:
        self.time += 1
        vectors = [_normalize(x) for x in embeddings]
        if not vectors:
            return []

        result: list[IdentityToken | None] = [None] * len(vectors)
        used_committed: set[int] = set()

        committed_ids, committed_proto = self._committed_prototypes()
        if committed_ids:
            cost = np.asarray(
                [
                    [cosine_distance(x, p) for p in committed_proto]
                    for x in vectors
                ],
                dtype=float,
            )
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows.tolist(), cols.tolist()):
                if cost[row, col] <= self.accept_distance:
                    identity = committed_ids[col]
                    self.committed[identity].add(
                        vectors[row], self.time, self.max_samples
                    )
                    result[row] = IdentityToken(False, identity)
                    used_committed.add(identity)

        unmatched = [i for i, token in enumerate(result) if token is None]
        provisional_ids, provisional_proto = self._active_provisionals()
        provisional_assignment: dict[int, int] = {}

        if unmatched and provisional_ids:
            cost = np.asarray(
                [
                    [
                        cosine_distance(vectors[index], prototype)
                        for prototype in provisional_proto
                    ]
                    for index in unmatched
                ],
                dtype=float,
            )
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows.tolist(), cols.tolist()):
                if cost[row, col] <= self.provisional_distance:
                    provisional_assignment[unmatched[row]] = provisional_ids[col]

        for index in unmatched:
            if index in provisional_assignment:
                provisional_id = provisional_assignment[index]
                cluster = self.provisional[provisional_id]
                cluster.add(vectors[index], self.time, self.max_samples)
            else:
                provisional_id = self.next_provisional
                self.next_provisional += 1
                cluster = _Cluster([vectors[index]], self.time)
                self.provisional[provisional_id] = cluster

            if cluster.support < self.commit_support:
                result[index] = IdentityToken(True, provisional_id)
                continue

            prototype = cluster.prototype()
            candidates = [
                (cosine_distance(prototype, entity.prototype()), identity)
                for identity, entity in self.committed.items()
                if identity not in used_committed
            ]
            best = min(candidates, default=(float("inf"), -1))
            if best[0] <= self.reconcile_distance:
                identity = best[1]
                for sample in cluster.samples:
                    self.committed[identity].add(
                        sample, self.time, self.max_samples
                    )
                self.reconciled[provisional_id] = identity
                self.reconciliations += 1
            else:
                if len(self.committed) >= self.max_entities:
                    result[index] = IdentityToken(True, provisional_id)
                    continue
                identity = self.next_committed
                self.next_committed += 1
                self.committed[identity] = _Cluster(
                    list(cluster.samples),
                    self.time,
                )
                self.reconciled[provisional_id] = identity
                self.new_commits += 1

            used_committed.add(identity)
            result[index] = IdentityToken(False, identity)

        return [token for token in result if token is not None]

    def resolve(self, token: IdentityToken) -> int | None:
        if not token.provisional:
            return token.identity
        return self.reconciled.get(token.identity)

    @property
    def committed_entities(self) -> int:
        return len(self.committed)


def identity_metrics(
    records: Sequence[tuple[int, int | None]],
    *,
    truth_count: int,
) -> dict[str, float]:
    by_identity: dict[int, dict[int, int]] = {}
    by_truth: dict[int, set[int]] = {}
    resolved = 0
    for truth, identity in records:
        if identity is None:
            continue
        resolved += 1
        counts = by_identity.setdefault(identity, {})
        counts[truth] = counts.get(truth, 0) + 1
        by_truth.setdefault(truth, set()).add(identity)

    if resolved == 0:
        return {
            "purity": 0.0,
            "fragmentation": 0.0,
            "coverage": 0.0,
            "identities": 0.0,
        }

    correct = sum(max(counts.values()) for counts in by_identity.values())
    return {
        "purity": correct / resolved,
        "fragmentation": float(
            np.mean(
                [
                    len(by_truth.get(truth, set()))
                    for truth in range(truth_count)
                ]
            )
        ),
        "coverage": resolved / len(records),
        "identities": float(len(by_identity)),
    }
