"""Integration from Cortex-discovered latent concepts to semantic grounding.

This research-only bridge keeps ontology formation and naming causally
separated:

1. latent concepts are induced from predictive counts only;
2. entity-level perceptual/relational evidence is aggregated onto the active
   latent concept IDs after induction;
3. semantic labels are grounded over those IDs;
4. split/merge/retract lifecycle events generate semantic-lineage records.

Labels never enter latent concept induction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from latent_concept_induction import (
    ConceptLifecycleEvent,
    CortexLatentConceptLifecycle,
    LatentConcept,
)
from semantic_grounding import (
    CortexSemanticGrounder,
    NeuralLabelEvidence,
    PerceptionFeedback,
    SemanticEdge,
    SemanticRead,
    StructuralSemanticObservation,
)

Entity = int
Label = str
Relation = str


@dataclass(frozen=True)
class EntityNeuralEvidence:
    entity: Entity
    probabilities: Mapping[Label, float]
    reliability: float = 1.0


@dataclass(frozen=True)
class EntityStructuralObservation:
    source: Entity
    relation: Relation
    target: Entity
    present: bool
    reliability: float = 1.0


@dataclass(frozen=True)
class IntegratedSemanticRead:
    concept_id: str
    members: frozenset[int]
    label: Label | None
    resolved: bool
    confidence: float
    margin: float
    neural_top_label: Label | None
    neural_top_confidence: float
    posterior: tuple[tuple[Label, float], ...]
    feedback: PerceptionFeedback


@dataclass(frozen=True)
class SemanticLineageEvent:
    epoch: int
    kind: str
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    source_labels: tuple[Label | None, ...]
    target_labels: tuple[Label | None, ...]


@dataclass(frozen=True)
class ConceptSemanticBridgeRead:
    epoch: int
    active_concepts: tuple[LatentConcept, ...]
    semantics: tuple[IntegratedSemanticRead, ...]
    concept_events: tuple[ConceptLifecycleEvent, ...]
    semantic_lineage: tuple[SemanticLineageEvent, ...]

    def semantic_by_id(self) -> dict[str, IntegratedSemanticRead]:
        return {
            read.concept_id: read
            for read in self.semantics
        }


def _entity_to_concept(
    active: Sequence[LatentConcept],
) -> dict[int, str]:
    result: dict[int, str] = {}
    for concept in active:
        for entity in concept.members:
            if entity in result:
                raise ValueError(
                    "active latent concepts overlap"
                )
            result[entity] = concept.concept_id
    return result


def _ground_current(
    active: Sequence[LatentConcept],
    labels: Sequence[Label],
    relations: Sequence[Relation],
    semantic_edges: Iterable[SemanticEdge],
    neural: Iterable[EntityNeuralEvidence],
    structural: Iterable[EntityStructuralObservation],
    *,
    accept_threshold: float,
    margin_threshold: float,
) -> tuple[IntegratedSemanticRead, ...]:
    active = tuple(active)
    if not active:
        return tuple()

    concepts = tuple(
        concept.concept_id
        for concept in active
    )
    membership = _entity_to_concept(active)

    grounder = CortexSemanticGrounder(
        concepts,
        labels,
        relations,
        semantic_edges,
        accept_threshold=accept_threshold,
        margin_threshold=margin_threshold,
    )

    for item in neural:
        concept_id = membership.get(item.entity)
        if concept_id is None:
            continue
        grounder.observe_neural(
            NeuralLabelEvidence(
                concept=concept_id,
                probabilities=item.probabilities,
                reliability=item.reliability,
            )
        )

    for item in structural:
        source = membership.get(item.source)
        target = membership.get(item.target)
        if source is None or target is None:
            continue
        if source == target:
            continue
        grounder.observe_structure(
            StructuralSemanticObservation(
                source=source,
                relation=item.relation,
                target=target,
                present=item.present,
                reliability=item.reliability,
            )
        )

    members_by_id = {
        concept.concept_id: concept.members
        for concept in active
    }

    return tuple(
        IntegratedSemanticRead(
            concept_id=read.concept,
            members=members_by_id[read.concept],
            label=read.label,
            resolved=read.resolved,
            confidence=read.confidence,
            margin=read.margin,
            neural_top_label=read.neural_top_label,
            neural_top_confidence=read.neural_top_confidence,
            posterior=read.posterior,
            feedback=read.feedback,
        )
        for read in grounder.read_all()
    )


class CortexConceptSemanticBridge:
    """Compose latent ontology induction with post-hoc semantic grounding."""

    def __init__(
        self,
        entity_count: int,
        labels: Sequence[Label],
        relations: Sequence[Relation],
        semantic_edges: Iterable[SemanticEdge],
        *,
        alpha: float = 0.5,
        tie_tolerance: float = 1.0e-9,
        accept_threshold: float = 0.70,
        margin_threshold: float = 0.15,
    ) -> None:
        if entity_count < 1:
            raise ValueError("entity_count must be positive")
        if not labels:
            raise ValueError("labels must be non-empty")
        self.labels = tuple(labels)
        self.relations = tuple(relations)
        self.semantic_edges = tuple(semantic_edges)
        self.accept_threshold = float(accept_threshold)
        self.margin_threshold = float(margin_threshold)
        self.lifecycle = CortexLatentConceptLifecycle(
            entity_count,
            alpha=alpha,
            tie_tolerance=tie_tolerance,
        )
        self._previous_semantics: dict[
            str, IntegratedSemanticRead
        ] = {}

    def advance(
        self,
        epoch: int,
        counts: np.ndarray,
        *,
        neural: Iterable[EntityNeuralEvidence] = (),
        structural: Iterable[
            EntityStructuralObservation
        ] = (),
    ) -> ConceptSemanticBridgeRead:
        concept_read = self.lifecycle.advance(
            epoch,
            counts,
        )
        semantics = _ground_current(
            concept_read.active_concepts,
            self.labels,
            self.relations,
            self.semantic_edges,
            tuple(neural),
            tuple(structural),
            accept_threshold=self.accept_threshold,
            margin_threshold=self.margin_threshold,
        )
        current = {
            read.concept_id: read
            for read in semantics
        }

        lineage = tuple(
            SemanticLineageEvent(
                epoch=epoch,
                kind=event.kind,
                source_ids=event.source_ids,
                target_ids=event.target_ids,
                source_labels=tuple(
                    self._previous_semantics[source].label
                    if source in self._previous_semantics
                    else None
                    for source in event.source_ids
                ),
                target_labels=tuple(
                    current[target].label
                    if target in current
                    else None
                    for target in event.target_ids
                ),
            )
            for event in concept_read.events
        )

        self._previous_semantics = current

        return ConceptSemanticBridgeRead(
            epoch=epoch,
            active_concepts=concept_read.active_concepts,
            semantics=semantics,
            concept_events=concept_read.events,
            semantic_lineage=lineage,
        )
