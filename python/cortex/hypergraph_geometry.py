"""Soft relational hypergeometry for Cortex concept basins.

The hypergraph is a read-only geometric interpretation of evidence already
stored by Cortex. It never promotes a relation into a hard semantic rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import isfinite
from typing import Iterable


EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class RelationalHyperedge:
    edge_id: str
    subject_id: int
    property_id: int
    object_id: int
    subject: str
    property: str
    object: str
    support: float

    def __post_init__(self) -> None:
        if not self.edge_id:
            raise ValueError("edge_id must be non-empty")
        if not 0.0 <= self.support <= 1.0 or not isfinite(self.support):
            raise ValueError("hyperedge support must be finite and in [0,1]")

    @property
    def participants(self) -> tuple[str, str, str]:
        return (self.subject, self.property, self.object)


@dataclass(frozen=True, slots=True)
class BasinAdjacency:
    basin: str
    direct_support: float
    bottleneck_affinity: float
    resolution_distance: float
    shared_hyperedges: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RelationalDirection:
    relation: str
    source: str
    target: str
    support: float
    normalized_support: float
    edge_id: str
    orientation: str


@dataclass(frozen=True, slots=True)
class Hyperpath:
    source: str
    target: str
    concepts: tuple[str, ...]
    hyperedges: tuple[str, ...]
    bottleneck_affinity: float

    @property
    def resolution_distance(self) -> float:
        return 1.0 - self.bottleneck_affinity


@dataclass(frozen=True, slots=True)
class HypergraphLocalGeometry:
    anchor: str
    resolution: float
    incident_hyperedges: tuple[RelationalHyperedge, ...]
    adjacent_basins: tuple[BasinAdjacency, ...]
    directions: tuple[RelationalDirection, ...]
    articulated_component: tuple[str, ...]


class HypergraphGeometry:
    """Resolution-aware soft hypergraph over Cortex concept basins.

    Each statement is one weighted higher-order edge joining its subject,
    property, and object basins. The property therefore remains a concept basin
    while simultaneously inducing local relational directions.
    """

    def __init__(self, hyperedges: Iterable[RelationalHyperedge]):
        self.hyperedges = tuple(hyperedges)
        self._by_id = {edge.edge_id: edge for edge in self.hyperedges}
        if len(self._by_id) != len(self.hyperedges):
            raise ValueError("hyperedge ids must be unique")

        incident: dict[str, list[RelationalHyperedge]] = {}
        concepts: set[str] = set()
        for edge in self.hyperedges:
            for participant in edge.participants:
                incident.setdefault(participant, []).append(edge)
                concepts.add(participant)
        self._incident = {
            concept: tuple(
                sorted(rows, key=lambda row: (-row.support, row.edge_id))
            )
            for concept, rows in incident.items()
        }
        self.concepts = tuple(sorted(concepts))

    def incident(
        self,
        concept: str,
        *,
        resolution: float = 0.0,
    ) -> tuple[RelationalHyperedge, ...]:
        self._validate_resolution(resolution)
        return tuple(
            edge
            for edge in self._incident.get(concept, ())
            if edge.support + EPS >= resolution
        )

    def direct_neighbors(
        self,
        concept: str,
        *,
        resolution: float = 0.0,
    ) -> tuple[BasinAdjacency, ...]:
        self._validate_resolution(resolution)
        shared: dict[str, list[RelationalHyperedge]] = {}
        for edge in self.incident(concept, resolution=resolution):
            for participant in edge.participants:
                if participant == concept:
                    continue
                shared.setdefault(participant, []).append(edge)

        rows: list[BasinAdjacency] = []
        for neighbor, edges in shared.items():
            direct = max(edge.support for edge in edges)
            affinity = self.bottleneck_affinity(
                concept,
                neighbor,
                resolution=resolution,
            )
            rows.append(
                BasinAdjacency(
                    basin=neighbor,
                    direct_support=direct,
                    bottleneck_affinity=affinity,
                    resolution_distance=1.0 - affinity,
                    shared_hyperedges=tuple(edge.edge_id for edge in edges),
                )
            )
        rows.sort(
            key=lambda row: (
                -row.direct_support,
                -row.bottleneck_affinity,
                row.basin,
            )
        )
        return tuple(rows)

    def directions(
        self,
        concept: str,
        *,
        resolution: float = 0.0,
    ) -> tuple[RelationalDirection, ...]:
        """Return local role-aware directions through relation basins.

        Subject points expose forward relation->object directions.
        Object points expose reverse relation->subject directions.
        Property points expose the observed subject->object examples that shape
        that relation basin.
        """

        self._validate_resolution(resolution)
        raw: list[tuple[str, str, str, float, str, str]] = []
        for edge in self.incident(concept, resolution=resolution):
            if concept == edge.subject:
                raw.append(
                    (
                        edge.property,
                        edge.subject,
                        edge.object,
                        edge.support,
                        edge.edge_id,
                        "forward",
                    )
                )
            if concept == edge.object:
                raw.append(
                    (
                        edge.property,
                        edge.object,
                        edge.subject,
                        edge.support,
                        edge.edge_id,
                        "reverse",
                    )
                )
            if concept == edge.property:
                raw.append(
                    (
                        edge.property,
                        edge.subject,
                        edge.object,
                        edge.support,
                        edge.edge_id,
                        "relation-example",
                    )
                )

        totals: dict[tuple[str, str, str], float] = {}
        for relation, source, _target, support, _edge_id, orientation in raw:
            key = (relation, source, orientation)
            totals[key] = totals.get(key, 0.0) + support

        out = tuple(
            RelationalDirection(
                relation=relation,
                source=source,
                target=target,
                support=support,
                normalized_support=(
                    support
                    / max(EPS, totals[(relation, source, orientation)])
                ),
                edge_id=edge_id,
                orientation=orientation,
            )
            for relation, source, target, support, edge_id, orientation in raw
        )
        return tuple(
            sorted(
                out,
                key=lambda row: (
                    -row.support,
                    row.relation,
                    row.source,
                    row.target,
                    row.edge_id,
                ),
            )
        )

    def transport(
        self,
        relation: str,
        source: str,
        *,
        resolution: float = 0.0,
    ) -> tuple[RelationalDirection, ...]:
        """Evaluate the learned local relation field from source to targets."""

        self._validate_resolution(resolution)
        rows = [
            edge
            for edge in self._incident.get(source, ())
            if (
                edge.subject == source
                and edge.property == relation
                and edge.support + EPS >= resolution
            )
        ]
        total = sum(edge.support for edge in rows)
        return tuple(
            RelationalDirection(
                relation=relation,
                source=source,
                target=edge.object,
                support=edge.support,
                normalized_support=edge.support / max(EPS, total),
                edge_id=edge.edge_id,
                orientation="forward",
            )
            for edge in sorted(
                rows,
                key=lambda edge: (-edge.support, edge.object, edge.edge_id),
            )
        )

    def articulated_component(
        self,
        concept: str,
        *,
        resolution: float,
    ) -> tuple[str, ...]:
        """H0-style component visible at the requested evidence resolution."""

        self._validate_resolution(resolution)
        if concept not in self._incident:
            return (concept,)

        seen = {concept}
        frontier = [concept]
        while frontier:
            current = frontier.pop()
            for edge in self.incident(current, resolution=resolution):
                for participant in edge.participants:
                    if participant not in seen:
                        seen.add(participant)
                        frontier.append(participant)
        return tuple(sorted(seen))

    def bottleneck_affinity(
        self,
        source: str,
        target: str,
        *,
        resolution: float = 0.0,
    ) -> float:
        path = self.strongest_path(
            source,
            target,
            resolution=resolution,
        )
        return path.bottleneck_affinity if path is not None else 0.0

    def resolution_distance(
        self,
        source: str,
        target: str,
        *,
        resolution: float = 0.0,
    ) -> float:
        return 1.0 - self.bottleneck_affinity(
            source,
            target,
            resolution=resolution,
        )

    def strongest_path(
        self,
        source: str,
        target: str,
        *,
        resolution: float = 0.0,
    ) -> Hyperpath | None:
        """Maximum-bottleneck hyperpath, tie-broken by fewer hyperedges."""

        self._validate_resolution(resolution)
        if source == target:
            return Hyperpath(source, target, (source,), (), 1.0)
        if source not in self._incident or target not in self._incident:
            return None

        best: dict[str, tuple[float, int]] = {source: (1.0, 0)}
        parent: dict[str, tuple[str, str]] = {}
        heap: list[tuple[float, int, str]] = [(-1.0, 0, source)]

        while heap:
            neg_capacity, hops, current = heapq.heappop(heap)
            capacity = -neg_capacity
            if best.get(current) != (capacity, hops):
                continue
            if current == target:
                break

            for edge in self.incident(current, resolution=resolution):
                next_capacity = min(capacity, edge.support)
                for participant in edge.participants:
                    if participant == current:
                        continue
                    candidate = (next_capacity, hops + 1)
                    previous = best.get(participant)
                    better = (
                        previous is None
                        or candidate[0] > previous[0] + EPS
                        or (
                            abs(candidate[0] - previous[0]) <= EPS
                            and candidate[1] < previous[1]
                        )
                    )
                    if not better:
                        continue
                    best[participant] = candidate
                    parent[participant] = (current, edge.edge_id)
                    heapq.heappush(
                        heap,
                        (-candidate[0], candidate[1], participant),
                    )

        if target not in best:
            return None

        concepts = [target]
        edges = []
        current = target
        while current != source:
            previous, edge_id = parent[current]
            edges.append(edge_id)
            concepts.append(previous)
            current = previous
        concepts.reverse()
        edges.reverse()
        return Hyperpath(
            source=source,
            target=target,
            concepts=tuple(concepts),
            hyperedges=tuple(edges),
            bottleneck_affinity=best[target][0],
        )

    def local_geometry(
        self,
        anchor: str,
        *,
        resolution: float = 0.0,
        max_neighbors: int = 32,
    ) -> HypergraphLocalGeometry:
        self._validate_resolution(resolution)
        incident = self.incident(anchor, resolution=resolution)
        neighbors = self.direct_neighbors(anchor, resolution=resolution)[
            :max_neighbors
        ]
        directions = self.directions(anchor, resolution=resolution)
        component = self.articulated_component(
            anchor,
            resolution=resolution,
        )
        return HypergraphLocalGeometry(
            anchor=anchor,
            resolution=resolution,
            incident_hyperedges=incident,
            adjacent_basins=neighbors,
            directions=directions,
            articulated_component=component,
        )

    @staticmethod
    def _validate_resolution(resolution: float) -> None:
        if not isfinite(resolution) or not 0.0 <= resolution <= 1.0:
            raise ValueError("resolution must be finite and in [0,1]")
