"""Structural counterfactual reasoning prototype for Cortex Reasoning Map v1.

This research module mirrors the stable Cortex intervention/control evidence
thresholds, then adds conservative reachability semantics and ephemeral graph
edits. It tests structural causal influence, not full unit-level SCM
counterfactual inference.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from json import dumps
from typing import Iterable, Mapping, Sequence

Node = str
Edge = tuple[Node, Node]


@dataclass(frozen=True)
class EvidenceConfig:
    causal_min_do: int = 8
    causal_min_control: int = 8
    causal_positive_diff: float = 0.24
    causal_null_abs_diff: float = 0.10


@dataclass(frozen=True)
class CausalCell:
    do_a: float = 1.0
    do_b: float = 1.0
    ctrl_a: float = 1.0
    ctrl_b: float = 1.0
    do_n: int = 0
    ctrl_n: int = 0
    state: int = 0

    @property
    def p_do(self) -> float:
        return self.do_a / (self.do_a + self.do_b)

    @property
    def p_ctrl(self) -> float:
        return self.ctrl_a / (self.ctrl_a + self.ctrl_b)

    @property
    def effect(self) -> float:
        return self.p_do - self.p_ctrl


@dataclass(frozen=True)
class InfluenceQuery:
    source: Node
    target: Node
    state: int
    path: tuple[Node, ...]
    confirmed_reachable: bool
    possible_reachable: bool
    removed_edges: tuple[Edge, ...]

    @property
    def label(self) -> str:
        return {
            1: "affected",
            0: "unresolved",
            -1: "unaffected",
        }[self.state]


def _derive_state(cfg: EvidenceConfig, cell: CausalCell) -> int:
    if (
        cell.do_n < cfg.causal_min_do
        or cell.ctrl_n < cfg.causal_min_control
    ):
        return 0
    diff = cell.effect
    if diff >= cfg.causal_positive_diff:
        return 1
    if abs(diff) <= cfg.causal_null_abs_diff:
        return -1
    return 0


def _adjacency(edges: Iterable[Edge]) -> dict[Node, tuple[Node, ...]]:
    raw: dict[Node, set[Node]] = {}
    for source, target in edges:
        raw.setdefault(source, set()).add(target)
    return {
        source: tuple(sorted(targets))
        for source, targets in raw.items()
    }


def _path(
    source: Node,
    target: Node,
    edges: Iterable[Edge],
) -> tuple[Node, ...]:
    if source == target:
        return (source,)

    adjacency = _adjacency(edges)
    queue: list[Node] = [source]
    parent: dict[Node, Node | None] = {source: None}
    cursor = 0

    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for nxt in adjacency.get(node, ()):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == target:
                chain = [target]
                current = target
                while parent[current] is not None:
                    current = parent[current]  # type: ignore[index]
                    chain.append(current)
                chain.reverse()
                return tuple(chain)
            queue.append(nxt)

    return tuple()


class CortexStructuralCounterfactual:
    """Intervention-sensitive causal evidence plus ephemeral graph reasoning."""

    def __init__(
        self,
        nodes: Iterable[Node],
        *,
        config: EvidenceConfig | None = None,
    ) -> None:
        self.nodes = tuple(sorted(set(nodes)))
        if len(self.nodes) < 2:
            raise ValueError("at least two nodes are required")
        self.config = config or EvidenceConfig()
        self.cells: dict[Edge, CausalCell] = {}

    def observe_counts(
        self,
        source: Node,
        target: Node,
        *,
        do_positive: int,
        do_negative: int,
        ctrl_positive: int,
        ctrl_negative: int,
    ) -> None:
        if source == target:
            raise ValueError("self-causal evidence is not supported")
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("unknown node")
        counts = (
            do_positive,
            do_negative,
            ctrl_positive,
            ctrl_negative,
        )
        if any(value < 0 for value in counts):
            raise ValueError("evidence counts must be non-negative")

        previous = self.cells.get((source, target), CausalCell())
        updated = replace(
            previous,
            do_a=previous.do_a + do_positive,
            do_b=previous.do_b + do_negative,
            ctrl_a=previous.ctrl_a + ctrl_positive,
            ctrl_b=previous.ctrl_b + ctrl_negative,
            do_n=previous.do_n + do_positive + do_negative,
            ctrl_n=(
                previous.ctrl_n
                + ctrl_positive
                + ctrl_negative
            ),
        )
        updated = replace(
            updated,
            state=_derive_state(self.config, updated),
        )
        self.cells[(source, target)] = updated

    def causal(self, source: Node, target: Node) -> CausalCell | None:
        return self.cells.get((source, target))

    def direct_state(self, source: Node, target: Node) -> int:
        cell = self.causal(source, target)
        return 0 if cell is None else cell.state

    def confirmed_edges(self) -> set[Edge]:
        return {
            edge
            for edge, cell in self.cells.items()
            if cell.state == 1
        }

    def possible_edges(self) -> set[Edge]:
        return {
            edge
            for edge, cell in self.cells.items()
            if cell.state in (0, 1)
        }

    def fingerprint(self) -> str:
        payload = {
            "nodes": self.nodes,
            "config": {
                "causal_min_do": self.config.causal_min_do,
                "causal_min_control": self.config.causal_min_control,
                "causal_positive_diff": self.config.causal_positive_diff,
                "causal_null_abs_diff": self.config.causal_null_abs_diff,
            },
            "cells": [
                {
                    "source": source,
                    "target": target,
                    "do_a": cell.do_a,
                    "do_b": cell.do_b,
                    "ctrl_a": cell.ctrl_a,
                    "ctrl_b": cell.ctrl_b,
                    "do_n": cell.do_n,
                    "ctrl_n": cell.ctrl_n,
                    "state": cell.state,
                }
                for (source, target), cell
                in sorted(self.cells.items())
            ],
        }
        encoded = dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def query(
        self,
        source: Node,
        target: Node,
        *,
        remove_edges: Iterable[Edge] = (),
    ) -> InfluenceQuery:
        if source == target:
            raise ValueError("source and target must differ")
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("unknown node")

        removed = tuple(sorted(set(remove_edges)))
        removed_set = set(removed)

        confirmed = self.confirmed_edges() - removed_set
        possible = self.possible_edges() - removed_set

        confirmed_path = _path(source, target, confirmed)
        if confirmed_path:
            return InfluenceQuery(
                source=source,
                target=target,
                state=1,
                path=confirmed_path,
                confirmed_reachable=True,
                possible_reachable=True,
                removed_edges=removed,
            )

        possible_path = _path(source, target, possible)
        if possible_path:
            return InfluenceQuery(
                source=source,
                target=target,
                state=0,
                path=tuple(),
                confirmed_reachable=False,
                possible_reachable=True,
                removed_edges=removed,
            )

        return InfluenceQuery(
            source=source,
            target=target,
            state=-1,
            path=tuple(),
            confirmed_reachable=False,
            possible_reachable=False,
            removed_edges=removed,
        )


class DirectEdgeBaseline:
    """No compositional reasoning: only inspect the source-target cell."""

    def __init__(self, model: CortexStructuralCounterfactual) -> None:
        self.model = model

    def query(self, source: Node, target: Node) -> int:
        return self.model.direct_state(source, target)


class CorrelationBaseline:
    """Mistake high control outcome rate for direct causality."""

    def __init__(
        self,
        model: CortexStructuralCounterfactual,
        *,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.threshold = threshold

    def direct_state(self, source: Node, target: Node) -> int:
        cell = self.model.causal(source, target)
        if cell is None:
            return 0
        return 1 if cell.p_ctrl > self.threshold else -1


def path_is_valid(
    path: Sequence[Node],
    *,
    source: Node,
    target: Node,
    active_edges: Iterable[Edge],
    removed_edges: Iterable[Edge] = (),
) -> bool:
    if not path:
        return False
    if path[0] != source or path[-1] != target:
        return False

    active = set(active_edges) - set(removed_edges)
    return all(
        (left, right) in active
        for left, right in zip(path, path[1:])
    )


def ground_truth_reachable(
    source: Node,
    target: Node,
    edges: Iterable[Edge],
    *,
    remove_edges: Iterable[Edge] = (),
) -> bool:
    return bool(
        _path(
            source,
            target,
            set(edges) - set(remove_edges),
        )
    )


def shortest_path_length(
    source: Node,
    target: Node,
    edges: Iterable[Edge],
) -> int | None:
    path = _path(source, target, edges)
    if not path:
        return None
    return len(path) - 1
