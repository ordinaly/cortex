"""Held-out composition over Cortex relational hypergeometry.

Research-only. Rules are induced from repeated geometric motifs; relation names
are treated as opaque basin identities.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .hypergraph_geometry import HypergraphGeometry, RelationalHyperedge


EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class RelationStep:
    relation: str
    orientation: str

    def __post_init__(self) -> None:
        if self.orientation not in {"forward", "reverse"}:
            raise ValueError("orientation must be forward or reverse")


@dataclass(frozen=True, slots=True)
class CompositionRule:
    signature: tuple[RelationStep, ...]
    target_relation: str
    examples: int
    confidence: float
    mean_support: float


@dataclass(frozen=True, slots=True)
class InferenceCandidate:
    target: str
    score: float
    path_support: float
    rule_confidence: float
    rule_examples: int
    signature: tuple[RelationStep, ...]


@dataclass(frozen=True, slots=True)
class CompositionRead:
    source: str
    relation: str
    candidates: tuple[InferenceCandidate, ...]

    @property
    def resolved(self) -> bool:
        return bool(self.candidates)

    @property
    def top(self) -> InferenceCandidate | None:
        return self.candidates[0] if self.candidates else None


class HypergraphCompositionLearner:
    """Induce relation compositions from repeated hypergraph triangles.

    For each observed direct relation edge (s, r, o), the learner searches for
    alternate role-aware paths from s to o that do not use that same edge.
    Repeated path signatures become evidence that the path composes to r.
    """

    def __init__(
        self,
        geometry: HypergraphGeometry,
        *,
        path_length: int = 2,
        minimum_examples: int = 2,
        confidence_threshold: float = 0.75,
        resolution: float = 0.0,
    ) -> None:
        if path_length < 2:
            raise ValueError("path_length must be at least 2")
        if minimum_examples < 1:
            raise ValueError("minimum_examples must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1]")
        if not 0.0 <= resolution <= 1.0 or not isfinite(resolution):
            raise ValueError("resolution must be finite and in [0,1]")
        self.geometry = geometry
        self.path_length = int(path_length)
        self.minimum_examples = int(minimum_examples)
        self.confidence_threshold = float(confidence_threshold)
        self.resolution = float(resolution)
        self._rules: tuple[CompositionRule, ...] | None = None

    def _steps_from(
        self,
        node: str,
        *,
        exclude_edge_id: str | None = None,
    ) -> tuple[tuple[RelationStep, str, float, str], ...]:
        rows: list[tuple[RelationStep, str, float, str]] = []
        for edge in self.geometry.incident(node, resolution=self.resolution):
            if edge.edge_id == exclude_edge_id:
                continue
            if edge.subject == node:
                rows.append(
                    (
                        RelationStep(edge.property, "forward"),
                        edge.object,
                        edge.support,
                        edge.edge_id,
                    )
                )
            if edge.object == node:
                rows.append(
                    (
                        RelationStep(edge.property, "reverse"),
                        edge.subject,
                        edge.support,
                        edge.edge_id,
                    )
                )
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    row[0].relation,
                    row[0].orientation,
                    row[1],
                    row[3],
                ),
            )
        )

    def _paths(
        self,
        source: str,
        target: str | None = None,
        *,
        exclude_edge_id: str | None = None,
        signature: tuple[RelationStep, ...] | None = None,
    ) -> tuple[
        tuple[tuple[RelationStep, ...], str, float, tuple[str, ...]],
        ...,
    ]:
        if signature is not None and len(signature) != self.path_length:
            raise ValueError("signature length does not match path_length")

        frontier = [
            (source, tuple(), 1.0, tuple(), frozenset({source}))
        ]
        for depth in range(self.path_length):
            next_frontier = []
            for node, steps, support, edge_ids, seen in frontier:
                for step, next_node, edge_support, edge_id in self._steps_from(
                    node,
                    exclude_edge_id=exclude_edge_id,
                ):
                    if signature is not None and step != signature[depth]:
                        continue
                    if next_node in seen:
                        continue
                    next_frontier.append(
                        (
                            next_node,
                            steps + (step,),
                            min(support, edge_support),
                            edge_ids + (edge_id,),
                            seen | {next_node},
                        )
                    )
            frontier = next_frontier

        rows = [
            (steps, node, support, edge_ids)
            for node, steps, support, edge_ids, _seen in frontier
            if target is None or node == target
        ]
        rows.sort(
            key=lambda row: (
                tuple(
                    (step.relation, step.orientation)
                    for step in row[0]
                ),
                row[1],
                -row[2],
                row[3],
            )
        )
        return tuple(rows)

    def fit(self) -> tuple[CompositionRule, ...]:
        if self._rules is not None:
            return self._rules

        counts: dict[
            tuple[tuple[RelationStep, ...], str], int
        ] = defaultdict(int)
        support_sums: dict[
            tuple[tuple[RelationStep, ...], str], float
        ] = defaultdict(float)
        totals_by_signature: dict[
            tuple[RelationStep, ...], int
        ] = defaultdict(int)

        for edge in self.geometry.hyperedges:
            if edge.support + EPS < self.resolution:
                continue
            seen_signatures: set[tuple[RelationStep, ...]] = set()
            for signature, endpoint, path_support, _edge_ids in self._paths(
                edge.subject,
                edge.object,
                exclude_edge_id=edge.edge_id,
            ):
                if endpoint != edge.object or signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                key = (signature, edge.property)
                counts[key] += 1
                support_sums[key] += min(path_support, edge.support)
                totals_by_signature[signature] += 1

        rules: list[CompositionRule] = []
        for (signature, target_relation), examples in counts.items():
            total = totals_by_signature[signature]
            confidence = examples / max(1, total)
            if examples < self.minimum_examples:
                continue
            if confidence + EPS < self.confidence_threshold:
                continue
            rules.append(
                CompositionRule(
                    signature=signature,
                    target_relation=target_relation,
                    examples=examples,
                    confidence=confidence,
                    mean_support=support_sums[
                        (signature, target_relation)
                    ]
                    / examples,
                )
            )

        rules.sort(
            key=lambda row: (
                row.target_relation,
                -row.confidence,
                -row.examples,
                tuple(
                    (step.relation, step.orientation)
                    for step in row.signature
                ),
            )
        )
        self._rules = tuple(rules)
        return self._rules

    @property
    def rules(self) -> tuple[CompositionRule, ...]:
        return self.fit()

    def infer(
        self,
        source: str,
        relation: str,
    ) -> CompositionRead:
        best: dict[str, InferenceCandidate] = {}
        for rule in self.fit():
            if rule.target_relation != relation:
                continue
            for signature, endpoint, path_support, _edge_ids in self._paths(
                source,
                signature=rule.signature,
            ):
                evidence_factor = rule.examples / (rule.examples + 1.0)
                score = (
                    path_support
                    * rule.confidence
                    * evidence_factor
                )
                candidate = InferenceCandidate(
                    target=endpoint,
                    score=score,
                    path_support=path_support,
                    rule_confidence=rule.confidence,
                    rule_examples=rule.examples,
                    signature=signature,
                )
                previous = best.get(endpoint)
                if previous is None or candidate.score > previous.score + EPS:
                    best[endpoint] = candidate

        return CompositionRead(
            source=source,
            relation=relation,
            candidates=tuple(
                sorted(
                    best.values(),
                    key=lambda row: (
                        -row.score,
                        row.target,
                    ),
                )
            ),
        )


def relation_edges(
    geometry: HypergraphGeometry,
    relation: str,
) -> tuple[RelationalHyperedge, ...]:
    return tuple(
        edge
        for edge in geometry.hyperedges
        if edge.property == relation
    )
