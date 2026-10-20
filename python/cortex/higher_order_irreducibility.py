"""Research-only higher-order composition over learned distinction neurons.

A filled 2-simplex must represent more than the presence of its three edges.
For a probe triple sigma and two contexts x,y, define

    R_sigma(x,y) = [D_sigma(x,y) - D_boundary(x,y)]_+

where D_sigma is total-variation distance between synchronized joint responses
of all three probes and D_boundary is the supremum of the three pairwise-face
TV distances.  The triple score is the supremum of R_sigma over supported
context pairs.

This implements a conservative operational notion of three-way irreducibility.
It is not claimed to be a canonical information-theoretic synergy measure.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Hashable, Iterable, Mapping

from .distinction_connectivity import HystereticDistinctionGraph, ProbeEdge
from .probe_economy import Probe, ProbeEconomy


ProbeTriple = tuple[Probe, Probe, Probe]


def _tv(left: Mapping[tuple, float], right: Mapping[tuple, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


class JointDistinctionMemory:
    """Bounded synchronized response memory for higher-order composition."""

    def __init__(
        self,
        economy: ProbeEconomy,
        *,
        window: int = 256,
        min_samples: int = 32,
    ) -> None:
        if type(window) is not int or window < 1:
            raise ValueError("window must be a positive integer")
        if type(min_samples) is not int or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if min_samples > window:
            raise ValueError("min_samples must not exceed window")
        self.economy = economy
        self.window = window
        self.min_samples = min_samples
        self._rows: dict[Hashable, deque[tuple[Hashable, ...]]] = {
            context: deque(maxlen=window) for context in economy.contexts
        }

    def _ordered(self, probes: Iterable[Iterable], *, size: int | None = None) -> tuple[Probe, ...]:
        rows = tuple(self.economy._check_probe(probe) for probe in probes)
        if len(rows) != len(set(rows)):
            raise ValueError("probes must be unique")
        if size is not None and len(rows) != size:
            raise ValueError(f"expected exactly {size} probes")
        return tuple(sorted(rows, key=self.economy._index.__getitem__))

    def observe(self, context: Hashable, responses: Mapping[Probe, Hashable]) -> None:
        if context not in self._rows:
            raise ValueError("context lies outside the economy context set")
        missing = [probe for probe in self.economy.probes if probe not in responses]
        if missing:
            raise ValueError("joint observation must contain every declared probe")
        sample = tuple(responses[probe] for probe in self.economy.probes)
        self._rows[context].append(sample)

    def support(self, context: Hashable) -> int:
        if context not in self._rows:
            raise ValueError("unknown context")
        return len(self._rows[context])

    def distribution(self, context: Hashable, probes: Iterable[Iterable]) -> dict[tuple, float]:
        probes = self._ordered(probes)
        if not probes:
            raise ValueError("distribution requires at least one probe")
        rows = self._rows.get(context)
        if rows is None:
            raise ValueError("unknown context")
        if len(rows) < self.min_samples:
            return {}
        indices = tuple(self.economy._index[probe] for probe in probes)
        counts = Counter(tuple(sample[index] for index in indices) for sample in rows)
        total = len(rows)
        return {key: count / total for key, count in counts.items()}

    def distance(self, left: Hashable, right: Hashable, probes: Iterable[Iterable]) -> float:
        if min(self.support(left), self.support(right)) < self.min_samples:
            return 0.0
        return _tv(self.distribution(left, probes), self.distribution(right, probes))

    def boundary_distance(self, left: Hashable, right: Hashable, triple: Iterable[Iterable]) -> float:
        triple = self._ordered(triple, size=3)
        return max(
            self.distance(left, right, face)
            for face in combinations(triple, 2)
        )

    def residual(self, left: Hashable, right: Hashable, triple: Iterable[Iterable]) -> float:
        triple = self._ordered(triple, size=3)
        joint = self.distance(left, right, triple)
        boundary = self.boundary_distance(left, right, triple)
        return max(0.0, joint - boundary)

    def irreducibility(self, triple: Iterable[Iterable]) -> float:
        triple = self._ordered(triple, size=3)
        return max(
            (
                self.residual(left, right, triple)
                for left, right in combinations(self.economy.contexts, 2)
                if min(self.support(left), self.support(right)) >= self.min_samples
            ),
            default=0.0,
        )


@dataclass(frozen=True, slots=True)
class HigherOrderDistinctionPolicy:
    simplex_on: float = 0.25
    simplex_off: float = 0.15

    def __post_init__(self) -> None:
        for name, value in (("simplex_on", self.simplex_on), ("simplex_off", self.simplex_off)):
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0,1]")
        if self.simplex_off > self.simplex_on:
            raise ValueError("simplex_off must not exceed simplex_on")


class HystereticDistinctionComplex:
    """2-dimensional simplicial complex over the existing distinction graph."""

    def __init__(
        self,
        graph: HystereticDistinctionGraph,
        memory: JointDistinctionMemory,
        *,
        policy: HigherOrderDistinctionPolicy | None = None,
    ) -> None:
        if graph.connectivity.economy is not memory.economy:
            raise ValueError("graph and joint memory must share one ProbeEconomy")
        self.graph = graph
        self.memory = memory
        self.policy = policy or HigherOrderDistinctionPolicy()
        self.triangles: set[ProbeTriple] = set()
        self.activations = 0
        self.retirements = 0
        self.steps = 0

    def _triple(self, probes: Iterable[Iterable]) -> ProbeTriple:
        rows = self.memory._ordered(probes, size=3)
        return rows[0], rows[1], rows[2]

    def _faces(self, triple: ProbeTriple) -> tuple[ProbeEdge, ProbeEdge, ProbeEdge]:
        faces = []
        for left, right in combinations(triple, 2):
            faces.append(self.graph.connectivity._pair(left, right))
        return faces[0], faces[1], faces[2]

    def candidate_triangles(self) -> tuple[ProbeTriple, ...]:
        candidates = []
        for probes in combinations(self.graph.vertices, 3):
            triple = self._triple(probes)
            if all(face in self.graph.edges for face in self._faces(triple)):
                candidates.append(triple)
        return tuple(candidates)

    def step(self) -> tuple[ProbeTriple, ...]:
        self.steps += 1
        candidates = set(self.candidate_triangles())

        # Face closure is hard: losing any edge immediately removes its coface.
        for triple in tuple(self.triangles):
            if triple not in candidates:
                self.triangles.remove(triple)
                self.retirements += 1

        for triple in candidates:
            score = self.memory.irreducibility(triple)
            if triple in self.triangles:
                if score < self.policy.simplex_off:
                    self.triangles.remove(triple)
                    self.retirements += 1
            elif score >= self.policy.simplex_on:
                self.triangles.add(triple)
                self.activations += 1

        return tuple(
            sorted(
                self.triangles,
                key=lambda triple: tuple(self.memory.economy._index[probe] for probe in triple),
            )
        )

    @property
    def weighted_triangles(self) -> tuple[tuple[ProbeTriple, float], ...]:
        return tuple((triple, self.memory.irreducibility(triple)) for triple in self.step())

    @property
    def snapshot(self) -> dict:
        return {
            "vertices": len(self.graph.vertices),
            "edges": len(self.graph.edges),
            "triangles": len(self.triangles),
            "triangle_indices": tuple(
                tuple(self.memory.economy._index[probe] for probe in triple)
                for triple in sorted(
                    self.triangles,
                    key=lambda row: tuple(self.memory.economy._index[probe] for probe in row),
                )
            ),
            "activations": self.activations,
            "retirements": self.retirements,
            "steps": self.steps,
        }
