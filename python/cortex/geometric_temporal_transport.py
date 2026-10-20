"""Research-only geometric temporal transport for integrated articulations.

Unlike the original state-ID transition table, this transport compares the
geometry carried by SimplicialIntegrationRead objects.  Continuity is measured
on shared low-dimensional marginals, so identity can persist even when a
higher-order coface appears, disappears, or changes its joint structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, isfinite
from typing import Hashable, Iterable, Mapping

from .probe_economy import Probe
from .simplicial_integration import SimplicialIntegrationRead


Distribution = dict[tuple[Hashable, ...], float]


def _tv(left: Mapping[tuple, float], right: Mapping[tuple, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


@dataclass(frozen=True, slots=True)
class GeometricTransportPolicy:
    max_dimension: int = 1
    dimension_weights: tuple[float, ...] = (0.60, 0.40)
    overlap_penalty: float = 0.20
    inverse_temperature: float = 8.0

    def __post_init__(self) -> None:
        if type(self.max_dimension) is not int or self.max_dimension < 0:
            raise ValueError("max_dimension must be a non-negative integer")
        if len(self.dimension_weights) <= self.max_dimension:
            raise ValueError("dimension_weights must cover every requested dimension")
        for weight in self.dimension_weights:
            if not isfinite(float(weight)) or float(weight) < 0.0:
                raise ValueError("dimension weights must be finite and non-negative")
        if sum(self.dimension_weights[: self.max_dimension + 1]) <= 0.0:
            raise ValueError("at least one used dimension weight must be positive")
        if not isfinite(self.overlap_penalty) or not 0.0 <= self.overlap_penalty <= 1.0:
            raise ValueError("overlap_penalty must lie in [0,1]")
        if not isfinite(self.inverse_temperature) or self.inverse_temperature <= 0.0:
            raise ValueError("inverse_temperature must be finite and positive")


@dataclass(frozen=True, slots=True)
class TransportComparison:
    distance: float
    overlap: float
    by_dimension: tuple[tuple[int, float], ...]
    topology_change: float
    full_joint_distance: float
    shared_probes: tuple[Probe, ...]


class GeometricTemporalTransport:
    """Transport kernel K over articulation geometry rather than state IDs."""

    def __init__(self, *, policy: GeometricTransportPolicy | None = None) -> None:
        self.policy = policy or GeometricTransportPolicy()

    @staticmethod
    def _marginal(
        read: SimplicialIntegrationRead,
        support: tuple[Probe, ...],
    ) -> Distribution:
        index = {probe: i for i, probe in enumerate(read.probes)}
        if any(probe not in index for probe in support):
            raise ValueError("marginal support is not present in articulation")
        positions = tuple(index[probe] for probe in support)
        result: Distribution = {}
        for state, probability in read.distribution:
            key = tuple(state[position] for position in positions)
            result[key] = result.get(key, 0.0) + probability
        return result

    @staticmethod
    def _constraint_supports(read: SimplicialIntegrationRead) -> frozenset[frozenset[Probe]]:
        supports = []
        for constraint in read.used_constraints:
            supports.append(frozenset(read.probes[index] for index in constraint))
        return frozenset(supports)

    def compare(
        self,
        source: SimplicialIntegrationRead,
        target: SimplicialIntegrationRead,
    ) -> TransportComparison:
        target_set = set(target.probes)
        shared = tuple(probe for probe in source.probes if probe in target_set)
        if not shared:
            raise ValueError("temporal transport requires at least one shared probe")
        union = set(source.probes) | set(target.probes)
        overlap = len(shared) / len(union)

        by_dimension: list[tuple[int, float]] = []
        weighted = 0.0
        weight_sum = 0.0
        upper = min(self.policy.max_dimension, len(shared) - 1)
        for dimension in range(upper + 1):
            rows = []
            for support in combinations(shared, dimension + 1):
                rows.append(_tv(self._marginal(source, support), self._marginal(target, support)))
            value = sum(rows) / len(rows)
            by_dimension.append((dimension, value))
            weight = self.policy.dimension_weights[dimension]
            weighted += weight * value
            weight_sum += weight

        geometry = weighted / weight_sum
        distance = min(1.0, geometry + self.policy.overlap_penalty * (1.0 - overlap))

        source_constraints = self._constraint_supports(source)
        target_constraints = self._constraint_supports(target)
        constraint_union = source_constraints | target_constraints
        topology_change = (
            len(source_constraints ^ target_constraints) / len(constraint_union)
            if constraint_union
            else 0.0
        )

        full_joint_distance = _tv(
            self._marginal(source, shared),
            self._marginal(target, shared),
        )
        return TransportComparison(
            distance=distance,
            overlap=overlap,
            by_dimension=tuple(by_dimension),
            topology_change=topology_change,
            full_joint_distance=full_joint_distance,
            shared_probes=shared,
        )

    def kernel(
        self,
        source: SimplicialIntegrationRead,
        candidates: Iterable[tuple[str, SimplicialIntegrationRead]],
    ) -> tuple[tuple[str, float], ...]:
        rows = tuple(candidates)
        if not rows:
            return ()
        labels = [label for label, _ in rows]
        if len(labels) != len(set(labels)):
            raise ValueError("candidate labels must be unique")
        scores = [
            exp(-self.policy.inverse_temperature * self.compare(source, target).distance)
            for _, target in rows
        ]
        total = sum(scores)
        return tuple((label, score / total) for (label, _), score in zip(rows, scores))

    def best_successor(
        self,
        source: SimplicialIntegrationRead,
        candidates: Iterable[tuple[str, SimplicialIntegrationRead]],
    ) -> tuple[str, float] | None:
        kernel = self.kernel(source, candidates)
        if not kernel:
            return None
        return min(kernel, key=lambda row: (-row[1], row[0]))
