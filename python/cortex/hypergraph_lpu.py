"""Exact LPU execution for Cortex-articulated relational compositions.

Cortex owns the learned relation geometry and composition rules. The LPU only
executes a temporary layered graph corresponding to one articulated rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .hypergraph_geometry import HypergraphGeometry, RelationalHyperedge
from .hypergraph_inference import (
    CompositionRule,
    InferenceCandidate,
    RelationStep,
)
from .lpu import LogicalProcessingUnit


EPS = 1.0e-12


@dataclass(frozen=True, slots=True)
class ExactRelationWitness:
    target: str
    layered_path: tuple[tuple[int, str], ...]
    hyperedge_ids: tuple[str, ...]
    exact: bool = True


@dataclass(frozen=True, slots=True)
class RuleExecutionRead:
    source: str
    relation: str
    candidates: tuple[InferenceCandidate, ...]
    witnesses: tuple[ExactRelationWitness, ...] = ()
    expanded_states: int = 0
    materialized_edges: int = 0
    completed_paths: int = 0
    lpu_calls: int = 0

    @property
    def resolved(self) -> bool:
        return bool(self.candidates)

    @property
    def top(self) -> InferenceCandidate | None:
        return self.candidates[0] if self.candidates else None


def _matching_edges(
    geometry: HypergraphGeometry,
    node: str,
    step: RelationStep,
    *,
    resolution: float,
) -> tuple[tuple[str, float, str], ...]:
    rows: list[tuple[str, float, str]] = []
    for edge in geometry.incident(node, resolution=resolution):
        if edge.property != step.relation:
            continue
        if step.orientation == "forward" and edge.subject == node:
            rows.append((edge.object, edge.support, edge.edge_id))
        elif step.orientation == "reverse" and edge.object == node:
            rows.append((edge.subject, edge.support, edge.edge_id))
    return tuple(
        sorted(
            rows,
            key=lambda row: (-row[1], row[0], row[2]),
        )
    )


def _candidate(
    target: str,
    path_support: float,
    rule: CompositionRule,
) -> InferenceCandidate:
    evidence_factor = rule.examples / (rule.examples + 1.0)
    return InferenceCandidate(
        target=target,
        score=path_support * rule.confidence * evidence_factor,
        path_support=path_support,
        rule_confidence=rule.confidence,
        rule_examples=rule.examples,
        signature=rule.signature,
    )


class EnumeratingRuleExecutor:
    """Exact Cortex-side reference executor that enumerates matching paths."""

    def __init__(
        self,
        geometry: HypergraphGeometry,
        *,
        resolution: float = 0.0,
    ) -> None:
        self.geometry = geometry
        self.resolution = float(resolution)

    def infer(
        self,
        source: str,
        relation: str,
        rules: Iterable[CompositionRule],
    ) -> RuleExecutionRead:
        best: dict[str, InferenceCandidate] = {}
        expanded_states = 0
        completed_paths = 0

        for rule in rules:
            if rule.target_relation != relation:
                continue
            frontier = [(source, 1.0, frozenset({source}))]
            for step in rule.signature:
                next_frontier = []
                for node, support, seen in frontier:
                    expanded_states += 1
                    for target, edge_support, _edge_id in _matching_edges(
                        self.geometry,
                        node,
                        step,
                        resolution=self.resolution,
                    ):
                        if target in seen:
                            continue
                        next_frontier.append(
                            (
                                target,
                                min(support, edge_support),
                                seen | {target},
                            )
                        )
                frontier = next_frontier

            completed_paths += len(frontier)
            for target, support, _seen in frontier:
                candidate = _candidate(target, support, rule)
                previous = best.get(target)
                if previous is None or candidate.score > previous.score + EPS:
                    best[target] = candidate

        return RuleExecutionRead(
            source=source,
            relation=relation,
            candidates=tuple(
                sorted(best.values(), key=lambda row: (-row.score, row.target))
            ),
            expanded_states=expanded_states,
            completed_paths=completed_paths,
        )


class LPURuleExecutor:
    """Execute Cortex-learned rules as exact layered reachability problems."""

    def __init__(
        self,
        geometry: HypergraphGeometry,
        *,
        resolution: float = 0.0,
        lpu=LogicalProcessingUnit,
    ) -> None:
        self.geometry = geometry
        self.resolution = float(resolution)
        self.lpu = lpu

    def _articulate(
        self,
        source: str,
        signature: tuple[RelationStep, ...],
    ):
        current: dict[str, float] = {source: 1.0}
        layered_edges: set[tuple[tuple[int, str], tuple[int, str]]] = set()
        transition_edge: dict[
            tuple[tuple[int, str], tuple[int, str]],
            tuple[float, str],
        ] = {}
        expanded_states = 0

        for depth, step in enumerate(signature):
            nxt: dict[str, float] = {}
            for node, incoming_support in sorted(current.items()):
                expanded_states += 1
                for target, edge_support, edge_id in _matching_edges(
                    self.geometry,
                    node,
                    step,
                    resolution=self.resolution,
                ):
                    path_support = min(incoming_support, edge_support)
                    previous_support = nxt.get(target, 0.0)
                    if path_support > previous_support:
                        nxt[target] = path_support

                    left = (depth, node)
                    right = (depth + 1, target)
                    layered_edges.add((left, right))
                    key = (left, right)
                    previous = transition_edge.get(key)
                    if (
                        previous is None
                        or edge_support > previous[0] + EPS
                        or (
                            abs(edge_support - previous[0]) <= EPS
                            and edge_id < previous[1]
                        )
                    ):
                        transition_edge[key] = (edge_support, edge_id)
            current = nxt

        return (
            tuple(sorted(layered_edges, key=repr)),
            transition_edge,
            current,
            expanded_states,
        )

    def infer(
        self,
        source: str,
        relation: str,
        rules: Iterable[CompositionRule],
    ) -> RuleExecutionRead:
        best: dict[str, InferenceCandidate] = {}
        witnesses: dict[str, ExactRelationWitness] = {}
        expanded_states = 0
        materialized_edges = 0
        lpu_calls = 0

        for rule in rules:
            if rule.target_relation != relation:
                continue
            (
                layered_edges,
                transition_edge,
                final_support,
                expanded,
            ) = self._articulate(source, rule.signature)
            expanded_states += expanded
            materialized_edges += len(layered_edges)

            start = (0, source)
            depth = len(rule.signature)
            for target, support in sorted(final_support.items()):
                goal = (depth, target)
                result = self.lpu.evaluate(
                    "REACHABLE",
                    layered_edges,
                    start,
                    goal,
                )
                lpu_calls += 1
                if result.value is not True:
                    continue

                layered_path = tuple(result.steps[0].result)
                hyperedge_ids = []
                valid = len(layered_path) == depth + 1
                for left, right in zip(layered_path, layered_path[1:]):
                    metadata = transition_edge.get((left, right))
                    if metadata is None:
                        valid = False
                        break
                    hyperedge_ids.append(metadata[1])
                if not valid:
                    raise RuntimeError("LPU returned an invalid layered witness")

                candidate = _candidate(target, support, rule)
                previous = best.get(target)
                if previous is None or candidate.score > previous.score + EPS:
                    best[target] = candidate
                    witnesses[target] = ExactRelationWitness(
                        target=target,
                        layered_path=layered_path,
                        hyperedge_ids=tuple(hyperedge_ids),
                    )

        ordered = tuple(
            sorted(best.values(), key=lambda row: (-row.score, row.target))
        )
        return RuleExecutionRead(
            source=source,
            relation=relation,
            candidates=ordered,
            witnesses=tuple(
                witnesses[row.target]
                for row in ordered
                if row.target in witnesses
            ),
            expanded_states=expanded_states,
            materialized_edges=materialized_edges,
            completed_paths=len(ordered),
            lpu_calls=lpu_calls,
        )


def raw_lpu_reachability(
    geometry: HypergraphGeometry,
    source: str,
    targets: Iterable[str],
    *,
    resolution: float = 0.0,
    lpu=LogicalProcessingUnit,
) -> dict[str, bool]:
    """Relation-agnostic LPU control.

    This intentionally discards relation identities. It demonstrates that exact
    graph reachability alone is not enough to select a semantically articulated
    relation path.
    """

    edges = tuple(
        sorted(
            {
                (edge.subject, edge.object)
                for edge in geometry.hyperedges
                if edge.support + EPS >= resolution
            },
            key=repr,
        )
    )
    return {
        target: bool(
            lpu.evaluate("REACHABLE", edges, source, target).value
        )
        for target in targets
    }
