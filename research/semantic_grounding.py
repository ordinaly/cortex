"""Research prototype for mutual neural/Cortex semantic grounding.

This module does not alter the stable native runtime. It models Stage-II-A
inference-time feedback between a perceptual label proposer and Cortex
structural reasoning over already-discovered concept identities.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from typing import Iterable, Mapping, Sequence

Concept = str
Label = str
Relation = str
SemanticEdge = tuple[Label, Relation, Label]


@dataclass(frozen=True)
class NeuralLabelEvidence:
    """One full candidate-label distribution emitted by perception."""

    concept: Concept
    probabilities: Mapping[Label, float]
    reliability: float = 1.0


@dataclass(frozen=True)
class StructuralSemanticObservation:
    """Observed structural fact about two internal concepts."""

    source: Concept
    relation: Relation
    target: Concept
    present: bool
    reliability: float = 1.0


@dataclass(frozen=True)
class PerceptionFeedback:
    """Cortex-to-perception inference-time feedback."""

    concept: Concept
    action: str
    promote_label: Label | None
    demote_label: Label | None
    strength: float
    reason: str


@dataclass(frozen=True)
class SemanticRead:
    concept: Concept
    label: Label | None
    resolved: bool
    confidence: float
    margin: float
    neural_top_label: Label | None
    neural_top_confidence: float
    posterior: tuple[tuple[Label, float], ...]
    feedback: PerceptionFeedback


class CortexSemanticGrounder:
    """Joint neural + structural label adjudication over a finite concept set.

    v1 deliberately assumes a flat one-to-one concept/label vocabulary. It
    enumerates the finite mapping space exactly, which makes uncertainty and
    provenance auditable. Later versions may replace enumeration with a sparse
    assignment/version-space approximation.
    """

    def __init__(
        self,
        concepts: Sequence[Concept],
        labels: Sequence[Label],
        relations: Sequence[Relation],
        semantic_edges: Iterable[SemanticEdge],
        *,
        neural_weight: float = 1.0,
        structural_weight: float = 1.0,
        structural_match_probability: float = 0.95,
        accept_threshold: float = 0.70,
        margin_threshold: float = 0.15,
        feedback_threshold: float = 0.12,
        epsilon: float = 1.0e-12,
    ) -> None:
        self.concepts = tuple(concepts)
        self.labels = tuple(labels)
        self.relations = tuple(relations)
        self.semantic_edges = frozenset(semantic_edges)

        if not self.concepts or not self.labels:
            raise ValueError("concepts and labels must be non-empty")
        if len(self.concepts) != len(self.labels):
            raise ValueError("semantic-grounding-v1 requires equal concept/label counts")
        if len(self.concepts) != len(set(self.concepts)):
            raise ValueError("concepts must be unique")
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("labels must be unique")
        if len(self.relations) != len(set(self.relations)):
            raise ValueError("relations must be unique")
        if neural_weight < 0.0 or structural_weight < 0.0:
            raise ValueError("evidence weights must be non-negative")
        if not 0.5 < structural_match_probability < 1.0:
            raise ValueError("structural_match_probability must be in (0.5, 1)")
        if not 0.0 < accept_threshold <= 1.0:
            raise ValueError("accept_threshold must be in (0, 1]")
        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError("margin_threshold must be in [0, 1]")
        if not 0.0 <= feedback_threshold <= 1.0:
            raise ValueError("feedback_threshold must be in [0, 1]")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        label_set = set(self.labels)
        relation_set = set(self.relations)
        for left, relation, right in self.semantic_edges:
            if left == right:
                raise ValueError("semantic self-edges are not supported in v1")
            if left not in label_set or right not in label_set:
                raise ValueError("semantic edge uses an unknown label")
            if relation not in relation_set:
                raise ValueError("semantic edge uses an unknown relation")

        self.neural_weight = float(neural_weight)
        self.structural_weight = float(structural_weight)
        self.structural_match_probability = float(structural_match_probability)
        self.accept_threshold = float(accept_threshold)
        self.margin_threshold = float(margin_threshold)
        self.feedback_threshold = float(feedback_threshold)
        self.epsilon = float(epsilon)

        self._neural: dict[Concept, list[NeuralLabelEvidence]] = {
            concept: [] for concept in self.concepts
        }
        self._structure: list[StructuralSemanticObservation] = []
        self._posterior_cache: tuple[tuple[tuple[Label, ...], float], ...] | None = None

    def observe_neural(self, evidence: NeuralLabelEvidence) -> None:
        if evidence.concept not in self._neural:
            raise ValueError("neural evidence uses an unknown concept")
        if not math.isfinite(evidence.reliability) or evidence.reliability <= 0.0:
            raise ValueError("neural reliability must be finite and positive")

        probabilities = dict(evidence.probabilities)
        if set(probabilities) != set(self.labels):
            raise ValueError("neural evidence must score every candidate label")
        values = list(probabilities.values())
        if any((not math.isfinite(p)) or p < 0.0 for p in values):
            raise ValueError("neural probabilities must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=1.0e-7, abs_tol=1.0e-7):
            raise ValueError("neural probabilities must sum to one")

        self._neural[evidence.concept].append(
            NeuralLabelEvidence(
                concept=evidence.concept,
                probabilities=probabilities,
                reliability=float(evidence.reliability),
            )
        )
        self._posterior_cache = None

    def observe_structure(self, observation: StructuralSemanticObservation) -> None:
        if observation.source == observation.target:
            raise ValueError("structural self-observations are not supported in v1")
        if observation.source not in self._neural or observation.target not in self._neural:
            raise ValueError("structural observation uses an unknown concept")
        if observation.relation not in self.relations:
            raise ValueError("structural observation uses an unknown relation")
        if not math.isfinite(observation.reliability) or observation.reliability <= 0.0:
            raise ValueError("structural reliability must be finite and positive")
        self._structure.append(observation)
        self._posterior_cache = None

    def observe_neural_many(self, evidence: Iterable[NeuralLabelEvidence]) -> None:
        for item in evidence:
            self.observe_neural(item)

    def observe_structure_many(
        self, observations: Iterable[StructuralSemanticObservation]
    ) -> None:
        for item in observations:
            self.observe_structure(item)

    def _neural_log_score(self, concept: Concept, label: Label) -> float:
        evidence = self._neural[concept]
        if not evidence:
            return -math.log(len(self.labels))
        return sum(
            item.reliability
            * math.log(max(self.epsilon, float(item.probabilities[label])))
            for item in evidence
        )

    def _mapping_log_score(self, label_targets: tuple[Label, ...]) -> float:
        mapping = dict(zip(self.concepts, label_targets))
        neural = sum(
            self._neural_log_score(concept, mapping[concept])
            for concept in self.concepts
        )

        q = self.structural_match_probability
        structural = 0.0
        for observation in self._structure:
            predicted = (
                mapping[observation.source],
                observation.relation,
                mapping[observation.target],
            ) in self.semantic_edges
            agrees = predicted == observation.present
            structural += observation.reliability * math.log(
                q if agrees else 1.0 - q
            )

        return self.neural_weight * neural + self.structural_weight * structural

    def mapping_posterior(self) -> tuple[tuple[tuple[Label, ...], float], ...]:
        """Return exact normalized support over one-to-one label mappings."""

        if self._posterior_cache is not None:
            return self._posterior_cache

        scored = [
            (tuple(label_targets), self._mapping_log_score(tuple(label_targets)))
            for label_targets in permutations(self.labels)
        ]
        maximum = max(score for _, score in scored)
        weights = [math.exp(score - maximum) for _, score in scored]
        total = sum(weights)
        self._posterior_cache = tuple(
            (targets, weight / total)
            for (targets, _), weight in zip(scored, weights)
        )
        return self._posterior_cache

    def label_marginals(self, concept: Concept) -> dict[Label, float]:
        if concept not in self._neural:
            raise ValueError("unknown concept")
        index = self.concepts.index(concept)
        marginals = {label: 0.0 for label in self.labels}
        for targets, weight in self.mapping_posterior():
            marginals[targets[index]] += weight
        return marginals

    def neural_mean_distribution(self, concept: Concept) -> dict[Label, float]:
        if concept not in self._neural:
            raise ValueError("unknown concept")
        evidence = self._neural[concept]
        if not evidence:
            uniform = 1.0 / len(self.labels)
            return {label: uniform for label in self.labels}
        total = sum(item.reliability for item in evidence)
        return {
            label: sum(
                item.reliability * float(item.probabilities[label])
                for item in evidence
            )
            / total
            for label in self.labels
        }

    def read(self, concept: Concept) -> SemanticRead:
        marginals = self.label_marginals(concept)
        ranked = tuple(
            sorted(marginals.items(), key=lambda item: (-item[1], item[0]))
        )
        best_label, confidence = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = confidence - runner_up

        neural = self.neural_mean_distribution(concept)
        neural_top_label, neural_top_confidence = max(
            neural.items(), key=lambda item: item[1]
        )

        resolved = (
            confidence >= self.accept_threshold
            and margin >= self.margin_threshold
        )
        label = best_label if resolved else None

        if resolved and best_label == neural_top_label:
            feedback = PerceptionFeedback(
                concept=concept,
                action="reinforce",
                promote_label=best_label,
                demote_label=None,
                strength=min(1.0, confidence * neural_top_confidence),
                reason="perception and structural posterior agree",
            )
        elif resolved:
            feedback = PerceptionFeedback(
                concept=concept,
                action="reinspect",
                promote_label=best_label,
                demote_label=neural_top_label,
                strength=min(
                    1.0,
                    margin
                    + max(
                        0.0,
                        confidence - neural.get(best_label, 0.0),
                    ),
                ),
                reason=(
                    "structural posterior promotes an alternative to the "
                    "neural top label"
                ),
            )
        elif (
            best_label != neural_top_label
            and confidence - neural.get(best_label, 0.0)
            >= self.feedback_threshold
        ):
            feedback = PerceptionFeedback(
                concept=concept,
                action="broaden",
                promote_label=best_label,
                demote_label=neural_top_label,
                strength=min(
                    1.0,
                    margin
                    + abs(confidence - neural.get(best_label, 0.0)),
                ),
                reason=(
                    "structural evidence conflicts with the neural ranking "
                    "but is not strong enough to relabel"
                ),
            )
        else:
            feedback = PerceptionFeedback(
                concept=concept,
                action="hold",
                promote_label=None,
                demote_label=None,
                strength=max(0.0, margin),
                reason="semantic evidence remains unresolved",
            )

        return SemanticRead(
            concept=concept,
            label=label,
            resolved=resolved,
            confidence=confidence,
            margin=margin,
            neural_top_label=neural_top_label,
            neural_top_confidence=neural_top_confidence,
            posterior=ranked,
            feedback=feedback,
        )

    def read_all(self) -> tuple[SemanticRead, ...]:
        return tuple(self.read(concept) for concept in self.concepts)
