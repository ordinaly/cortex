"""Learn pairwise topology between surviving distinguishability probes.

The v0.7 reference graph is built from *mutual non-redundancy*, not correlation.
For each probe q, Probe Economy already exposes a learner-visible set S_q of
context pairs that q currently vetoes merging. Let S be the full currently
required veto relation. For probes i,j define

    Gamma(i,j) = min(|S_i \\ S_j|, |S_j \\ S_i|) / |S|.

Gamma is zero if either probe contributes no separation structure beyond the
other. Positive Gamma means both probes add context distinctions that the other
cannot supply, providing a conservative reason to connect them.

This is a finite research topology over learned distinctions. It is not an
information-theoretic synergy theorem and it does not yet create higher
simplices.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isfinite
from typing import Iterable

from .probe_economy import HystereticProbePopulation, Probe, ProbeEconomy


ProbeEdge = tuple[Probe, Probe]


@dataclass(frozen=True, slots=True)
class DistinctionConnectivityPolicy:
    edge_on: float = 0.10
    edge_off: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (("edge_on", self.edge_on), ("edge_off", self.edge_off)):
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0,1]")
        if self.edge_off > self.edge_on:
            raise ValueError("edge_off must not exceed edge_on")


class DistinctionConnectivity:
    """Measure learner-visible mutual complementarity between probe neurons."""

    def __init__(self, economy: ProbeEconomy) -> None:
        self.economy = economy

    def _pair(self, left: Iterable, right: Iterable) -> ProbeEdge:
        left = self.economy._check_probe(left)
        right = self.economy._check_probe(right)
        if left == right:
            raise ValueError("connectivity requires two distinct probes")
        if self.economy._index[left] < self.economy._index[right]:
            return left, right
        return right, left

    def unique_edges(self, probe: Iterable, other: Iterable) -> frozenset:
        probe = self.economy._check_probe(probe)
        other = self.economy._check_probe(other)
        required = self.economy.required_edges
        return frozenset(
            (self.economy.veto_edges(probe) & required)
            - (self.economy.veto_edges(other) & required)
        )

    def complementarity(self, left: Iterable, right: Iterable) -> float:
        """Mutual non-redundant distinction coverage in [0,1]."""
        left, right = self._pair(left, right)
        required = self.economy.required_edges
        if not required:
            return 0.0
        left_unique = self.unique_edges(left, right)
        right_unique = self.unique_edges(right, left)
        return min(len(left_unique), len(right_unique)) / len(required)

    def joint_coverage_gain(self, left: Iterable, right: Iterable) -> float:
        """Equivalent union-over-best-single diagnostic for the current relation."""
        left, right = self._pair(left, right)
        required = self.economy.required_edges
        if not required:
            return 0.0
        left_edges = self.economy.veto_edges(left) & required
        right_edges = self.economy.veto_edges(right) & required
        union = left_edges | right_edges
        return (len(union) - max(len(left_edges), len(right_edges))) / len(required)

    def candidate_edges(self, probes: Iterable[Iterable]) -> tuple[ProbeEdge, ...]:
        rows = tuple(self.economy._check_probe(probe) for probe in probes)
        if len(rows) != len(set(rows)):
            raise ValueError("candidate probes must be unique")
        return tuple(self._pair(a, b) for a, b in combinations(rows, 2))


class HystereticDistinctionGraph:
    """Dynamic pairwise graph over the currently active distinction neurons."""

    def __init__(
        self,
        connectivity: DistinctionConnectivity,
        population: HystereticProbePopulation,
        *,
        policy: DistinctionConnectivityPolicy | None = None,
    ) -> None:
        if population.economy is not connectivity.economy:
            raise ValueError("connectivity and population must share one ProbeEconomy")
        self.connectivity = connectivity
        self.population = population
        self.policy = policy or DistinctionConnectivityPolicy()
        self.edges: set[ProbeEdge] = set()
        self.activations = 0
        self.retirements = 0
        self.steps = 0

    @property
    def vertices(self) -> tuple[Probe, ...]:
        return tuple(
            probe
            for probe in self.connectivity.economy.probes
            if probe in self.population.active
        )

    def _ordered_edges(self) -> tuple[ProbeEdge, ...]:
        return tuple(
            sorted(
                self.edges,
                key=lambda edge: (
                    self.connectivity.economy._index[edge[0]],
                    self.connectivity.economy._index[edge[1]],
                ),
            )
        )

    def step(self) -> tuple[ProbeEdge, ...]:
        self.steps += 1
        vertices = self.vertices
        vertex_set = set(vertices)

        # A representation vertex disappearing immediately removes its incident
        # topology. Hysteresis applies to relation evidence, not dead vertices.
        for edge in tuple(self.edges):
            if edge[0] not in vertex_set or edge[1] not in vertex_set:
                self.edges.remove(edge)
                self.retirements += 1

        for edge in self.connectivity.candidate_edges(vertices):
            score = self.connectivity.complementarity(*edge)
            if edge in self.edges:
                if score < self.policy.edge_off:
                    self.edges.remove(edge)
                    self.retirements += 1
            elif score >= self.policy.edge_on:
                self.edges.add(edge)
                self.activations += 1

        return self._ordered_edges()

    @property
    def weighted_edges(self) -> tuple[tuple[Probe, Probe, float], ...]:
        """Read current edge weights without mutating graph state."""
        return tuple(
            (left, right, self.connectivity.complementarity(left, right))
            for left, right in self._ordered_edges()
        )

    @property
    def snapshot(self) -> dict:
        return {
            "vertices": len(self.vertices),
            "edges": len(self.edges),
            "edge_indices": tuple(
                (
                    self.connectivity.economy._index[left],
                    self.connectivity.economy._index[right],
                )
                for left, right in self._ordered_edges()
            ),
            "activations": self.activations,
            "retirements": self.retirements,
            "steps": self.steps,
        }
