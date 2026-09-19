"""Structural analogy / isomorphism prototype for Cortex Reasoning Map v1.

The model keeps a version space of every joint entity/relation bijection that is
consistent with the target evidence observed so far. It resolves a held-out
structural fact only when all surviving correspondences agree.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Mapping, Sequence

Node = str
Relation = str
TypedEdge = tuple[Node, Relation, Node]


@dataclass(frozen=True)
class StructuralObservation:
    source: Node
    relation: Relation
    target: Node
    present: bool

    @property
    def triple(self) -> TypedEdge:
        return (self.source, self.relation, self.target)


@dataclass(frozen=True)
class CandidateMapping:
    node_targets: tuple[Node, ...]
    relation_targets: tuple[Relation, ...]
    mapped_edges: frozenset[TypedEdge]


@dataclass(frozen=True)
class AnalogyInference:
    value: bool | None
    resolved: bool
    incompatible: bool
    candidate_count: int
    unanimous_support: int
    support_fraction: float
    source_triples: tuple[TypedEdge, ...]


@dataclass(frozen=True)
class CorrespondenceRead:
    resolved: bool
    support: tuple[tuple[str, float], ...]
    candidate_count: int


class CortexStructuralAnalogy:
    """Finite typed-graph structural correspondence under partial evidence."""

    def __init__(
        self,
        source_nodes: Sequence[Node],
        source_relations: Sequence[Relation],
        source_edges: Iterable[TypedEdge],
        target_nodes: Sequence[Node],
        target_relations: Sequence[Relation],
    ) -> None:
        self.source_nodes = tuple(source_nodes)
        self.source_relations = tuple(source_relations)
        self.target_nodes = tuple(target_nodes)
        self.target_relations = tuple(target_relations)
        self.source_edges = frozenset(source_edges)

        if len(self.source_nodes) != len(set(self.source_nodes)):
            raise ValueError("source nodes must be unique")
        if len(self.target_nodes) != len(set(self.target_nodes)):
            raise ValueError("target nodes must be unique")
        if len(self.source_relations) != len(set(self.source_relations)):
            raise ValueError("source relations must be unique")
        if len(self.target_relations) != len(set(self.target_relations)):
            raise ValueError("target relations must be unique")
        if len(self.source_nodes) != len(self.target_nodes):
            raise ValueError("source/target node counts must match")
        if len(self.source_relations) != len(self.target_relations):
            raise ValueError("source/target relation counts must match")
        if not self.source_nodes or not self.source_relations:
            raise ValueError("non-empty node and relation sets are required")

        source_node_set = set(self.source_nodes)
        source_relation_set = set(self.source_relations)
        for left, relation, right in self.source_edges:
            if left == right:
                raise ValueError("self edges are not supported")
            if left not in source_node_set or right not in source_node_set:
                raise ValueError("source edge uses unknown node")
            if relation not in source_relation_set:
                raise ValueError("source edge uses unknown relation")

        self.observations: list[StructuralObservation] = []
        self._candidates = self._enumerate_candidates()

    def _enumerate_candidates(self) -> list[CandidateMapping]:
        candidates: list[CandidateMapping] = []
        for node_targets in permutations(self.target_nodes):
            node_map = dict(zip(self.source_nodes, node_targets))
            for relation_targets in permutations(self.target_relations):
                relation_map = dict(
                    zip(self.source_relations, relation_targets)
                )
                mapped = frozenset(
                    (
                        node_map[left],
                        relation_map[relation],
                        node_map[right],
                    )
                    for left, relation, right in self.source_edges
                )
                candidates.append(
                    CandidateMapping(
                        node_targets=tuple(node_targets),
                        relation_targets=tuple(relation_targets),
                        mapped_edges=mapped,
                    )
                )
        return candidates

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    @property
    def incompatible(self) -> bool:
        return not self._candidates

    @property
    def mapping_unique(self) -> bool:
        return len(self._candidates) == 1

    def _validate_target_triple(self, triple: TypedEdge) -> None:
        left, relation, right = triple
        if left == right:
            raise ValueError("self triples are not supported")
        if left not in self.target_nodes or right not in self.target_nodes:
            raise ValueError("unknown target node")
        if relation not in self.target_relations:
            raise ValueError("unknown target relation")

    def observe(self, observation: StructuralObservation) -> None:
        self._validate_target_triple(observation.triple)
        self.observations.append(observation)
        self._candidates = [
            candidate
            for candidate in self._candidates
            if (
                (observation.triple in candidate.mapped_edges)
                == observation.present
            )
        ]

    def observe_many(
        self,
        observations: Iterable[StructuralObservation],
    ) -> None:
        for observation in observations:
            self.observe(observation)

    def _inverse_maps(
        self,
        candidate: CandidateMapping,
    ) -> tuple[dict[Node, Node], dict[Relation, Relation]]:
        node_inverse = {
            target: source
            for source, target in zip(
                self.source_nodes,
                candidate.node_targets,
            )
        }
        relation_inverse = {
            target: source
            for source, target in zip(
                self.source_relations,
                candidate.relation_targets,
            )
        }
        return node_inverse, relation_inverse

    def infer(
        self,
        source: Node,
        relation: Relation,
        target: Node,
    ) -> AnalogyInference:
        triple = (source, relation, target)
        self._validate_target_triple(triple)

        if not self._candidates:
            return AnalogyInference(
                value=None,
                resolved=False,
                incompatible=True,
                candidate_count=0,
                unanimous_support=0,
                support_fraction=0.0,
                source_triples=tuple(),
            )

        predictions = [
            triple in candidate.mapped_edges
            for candidate in self._candidates
        ]
        positive = sum(predictions)
        negative = len(predictions) - positive
        unanimous = positive == 0 or negative == 0
        majority = max(positive, negative)
        value = predictions[0] if unanimous else None

        source_triples: set[TypedEdge] = set()
        if unanimous:
            for candidate in self._candidates:
                node_inverse, relation_inverse = self._inverse_maps(
                    candidate
                )
                source_triples.add(
                    (
                        node_inverse[source],
                        relation_inverse[relation],
                        node_inverse[target],
                    )
                )

        return AnalogyInference(
            value=value,
            resolved=unanimous,
            incompatible=False,
            candidate_count=len(self._candidates),
            unanimous_support=(
                len(self._candidates)
                if unanimous
                else majority
            ),
            support_fraction=majority / len(self._candidates),
            source_triples=tuple(sorted(source_triples)),
        )

    def entity_correspondence(
        self,
        source_node: Node,
    ) -> CorrespondenceRead:
        if source_node not in self.source_nodes:
            raise ValueError("unknown source node")
        if not self._candidates:
            return CorrespondenceRead(False, tuple(), 0)

        index = self.source_nodes.index(source_node)
        counts = Counter(
            candidate.node_targets[index]
            for candidate in self._candidates
        )
        total = len(self._candidates)
        support = tuple(
            sorted(
                (
                    target,
                    count / total,
                )
                for target, count in counts.items()
            )
        )
        return CorrespondenceRead(
            resolved=len(counts) == 1,
            support=support,
            candidate_count=total,
        )

    def relation_correspondence(
        self,
        source_relation: Relation,
    ) -> CorrespondenceRead:
        if source_relation not in self.source_relations:
            raise ValueError("unknown source relation")
        if not self._candidates:
            return CorrespondenceRead(False, tuple(), 0)

        index = self.source_relations.index(source_relation)
        counts = Counter(
            candidate.relation_targets[index]
            for candidate in self._candidates
        )
        total = len(self._candidates)
        support = tuple(
            sorted(
                (
                    target,
                    count / total,
                )
                for target, count in counts.items()
            )
        )
        return CorrespondenceRead(
            resolved=len(counts) == 1,
            support=support,
            candidate_count=total,
        )

    def resolved_entity_mapping(self) -> dict[Node, Node]:
        result = {}
        for source in self.source_nodes:
            read = self.entity_correspondence(source)
            if not read.resolved:
                continue
            result[source] = read.support[0][0]
        return result

    def resolved_relation_mapping(self) -> dict[Relation, Relation]:
        result = {}
        for source in self.source_relations:
            read = self.relation_correspondence(source)
            if not read.resolved:
                continue
            result[source] = read.support[0][0]
        return result


class DirectStructuralMemory:
    """Target-only memory baseline."""

    def __init__(
        self,
        observations: Iterable[StructuralObservation],
    ) -> None:
        self.observed = {
            observation.triple: observation.present
            for observation in observations
        }

    def infer(
        self,
        source: Node,
        relation: Relation,
        target: Node,
    ) -> AnalogyInference:
        triple = (source, relation, target)
        if triple not in self.observed:
            return AnalogyInference(
                value=None,
                resolved=False,
                incompatible=False,
                candidate_count=0,
                unanimous_support=0,
                support_fraction=0.0,
                source_triples=tuple(),
            )
        return AnalogyInference(
            value=self.observed[triple],
            resolved=True,
            incompatible=False,
            candidate_count=0,
            unanimous_support=1,
            support_fraction=1.0,
            source_triples=tuple(),
        )


def map_edges(
    source_edges: Iterable[TypedEdge],
    node_map: Mapping[Node, Node],
    relation_map: Mapping[Relation, Relation],
) -> frozenset[TypedEdge]:
    return frozenset(
        (
            node_map[left],
            relation_map[relation],
            node_map[right],
        )
        for left, relation, right in source_edges
    )


def all_target_triples(
    nodes: Sequence[Node],
    relations: Sequence[Relation],
) -> tuple[TypedEdge, ...]:
    return tuple(
        (left, relation, right)
        for left in nodes
        for right in nodes
        if left != right
        for relation in relations
    )


def full_observations(
    nodes: Sequence[Node],
    relations: Sequence[Relation],
    edges: Iterable[TypedEdge],
) -> tuple[StructuralObservation, ...]:
    edge_set = set(edges)
    return tuple(
        StructuralObservation(
            source=left,
            relation=relation,
            target=right,
            present=(left, relation, right) in edge_set,
        )
        for left, relation, right in all_target_triples(
            nodes,
            relations,
        )
    )
