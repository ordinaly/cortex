"""Temporal semantic revision for mutual neural/Cortex grounding.

Research-only Stage-II-A.2 prototype. It wraps the validated finite semantic
grounder with temporal evidence decay and explicit revision hysteresis.

The stable native Cortex runtime is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from semantic_grounding import (
    Concept,
    CortexSemanticGrounder,
    Label,
    NeuralLabelEvidence,
    Relation,
    SemanticEdge,
    SemanticRead,
    StructuralSemanticObservation,
)


@dataclass(frozen=True)
class TimedNeuralEvidence:
    epoch: int
    evidence: NeuralLabelEvidence


@dataclass(frozen=True)
class TimedStructuralEvidence:
    epoch: int
    evidence: StructuralSemanticObservation


@dataclass(frozen=True)
class RevisionEvent:
    epoch: int
    concept: Concept
    kind: str
    old_label: Label | None
    new_label: Label | None
    candidate_label: Label | None
    confidence: float
    margin: float
    reason: str


@dataclass(frozen=True)
class RevisionRead:
    epoch: int
    concept: Concept
    phase: str
    accepted_label: Label | None
    candidate_label: Label | None
    conflict_streak: int
    candidate_streak: int
    posterior_label: Label | None
    posterior_confidence: float
    posterior_margin: float
    neural_top_label: Label | None
    neural_top_confidence: float
    grounding: SemanticRead


@dataclass
class _MutableRevisionState:
    phase: str = "unresolved"
    accepted_label: Label | None = None
    previous_label: Label | None = None
    candidate_label: Label | None = None
    conflict_streak: int = 0
    candidate_streak: int = 0


class CortexSemanticRevision:
    """Maintain a revisable accepted semantic label over chronological evidence.

    Evidence is reconstructed into a fresh CortexSemanticGrounder at every
    epoch. Older evidence receives exponentially smaller reliability and is
    eventually removed by a finite evidence horizon.

    The accepted semantic state has its own hysteresis:

    stable -> challenged -> unresolved -> stable(new label)

    so the posterior may change before Cortex changes its accepted label.
    """

    def __init__(
        self,
        concepts: Sequence[Concept],
        labels: Sequence[Label],
        relations: Sequence[Relation],
        semantic_edges: Iterable[SemanticEdge],
        *,
        neural_decay: float = 0.72,
        structural_decay: float = 0.82,
        evidence_horizon: int = 12,
        challenge_patience: int = 2,
        revision_patience: int = 2,
        grounder_accept_threshold: float = 0.70,
        grounder_margin_threshold: float = 0.15,
    ) -> None:
        self.concepts = tuple(concepts)
        self.labels = tuple(labels)
        self.relations = tuple(relations)
        self.semantic_edges = frozenset(semantic_edges)

        if not 0.0 < neural_decay <= 1.0:
            raise ValueError("neural_decay must be in (0, 1]")
        if not 0.0 < structural_decay <= 1.0:
            raise ValueError("structural_decay must be in (0, 1]")
        if evidence_horizon < 1:
            raise ValueError("evidence_horizon must be positive")
        if challenge_patience < 1:
            raise ValueError("challenge_patience must be positive")
        if revision_patience < 1:
            raise ValueError("revision_patience must be positive")

        # Validate the declared grounding problem immediately.
        CortexSemanticGrounder(
            self.concepts,
            self.labels,
            self.relations,
            self.semantic_edges,
            accept_threshold=grounder_accept_threshold,
            margin_threshold=grounder_margin_threshold,
        )

        self.neural_decay = float(neural_decay)
        self.structural_decay = float(structural_decay)
        self.evidence_horizon = int(evidence_horizon)
        self.challenge_patience = int(challenge_patience)
        self.revision_patience = int(revision_patience)
        self.grounder_accept_threshold = float(grounder_accept_threshold)
        self.grounder_margin_threshold = float(grounder_margin_threshold)

        self.epoch = -1
        self._neural_history: list[TimedNeuralEvidence] = []
        self._structural_history: list[TimedStructuralEvidence] = []
        self._states = {
            concept: _MutableRevisionState()
            for concept in self.concepts
        }
        self._events: list[RevisionEvent] = []
        self._reads: dict[Concept, RevisionRead] = {}
        self._grounder: CortexSemanticGrounder | None = None

    @property
    def events(self) -> tuple[RevisionEvent, ...]:
        return tuple(self._events)

    @property
    def neural_history_size(self) -> int:
        return len(self._neural_history)

    @property
    def structural_history_size(self) -> int:
        return len(self._structural_history)

    def _prune(self) -> None:
        minimum = self.epoch - self.evidence_horizon + 1
        self._neural_history = [
            item for item in self._neural_history
            if item.epoch >= minimum
        ]
        self._structural_history = [
            item for item in self._structural_history
            if item.epoch >= minimum
        ]

    def _rebuild_grounder(self) -> CortexSemanticGrounder:
        grounder = CortexSemanticGrounder(
            self.concepts,
            self.labels,
            self.relations,
            self.semantic_edges,
            accept_threshold=self.grounder_accept_threshold,
            margin_threshold=self.grounder_margin_threshold,
        )

        for item in self._neural_history:
            age = self.epoch - item.epoch
            evidence = item.evidence
            grounder.observe_neural(
                NeuralLabelEvidence(
                    concept=evidence.concept,
                    probabilities=evidence.probabilities,
                    reliability=(
                        evidence.reliability
                        * (self.neural_decay ** age)
                    ),
                )
            )

        for item in self._structural_history:
            age = self.epoch - item.epoch
            evidence = item.evidence
            grounder.observe_structure(
                StructuralSemanticObservation(
                    source=evidence.source,
                    relation=evidence.relation,
                    target=evidence.target,
                    present=evidence.present,
                    reliability=(
                        evidence.reliability
                        * (self.structural_decay ** age)
                    ),
                )
            )

        return grounder

    def _emit(
        self,
        concept: Concept,
        kind: str,
        old_label: Label | None,
        new_label: Label | None,
        candidate_label: Label | None,
        read: SemanticRead,
        reason: str,
    ) -> None:
        self._events.append(
            RevisionEvent(
                epoch=self.epoch,
                concept=concept,
                kind=kind,
                old_label=old_label,
                new_label=new_label,
                candidate_label=candidate_label,
                confidence=read.confidence,
                margin=read.margin,
                reason=reason,
            )
        )

    def _posterior_candidate(self, read: SemanticRead) -> Label | None:
        if not read.posterior:
            return None
        return read.posterior[0][0]

    def _advance_state(
        self,
        concept: Concept,
        read: SemanticRead,
    ) -> None:
        state = self._states[concept]
        posterior_candidate = self._posterior_candidate(read)

        # Initial acceptance requires the normal grounding resolution rule.
        if state.accepted_label is None and state.previous_label is None:
            if read.resolved and read.label is not None:
                state.accepted_label = read.label
                state.phase = "stable"
                state.candidate_label = None
                state.candidate_streak = 0
                state.conflict_streak = 0
                self._emit(
                    concept,
                    "accepted",
                    None,
                    read.label,
                    read.label,
                    read,
                    "initial semantic posterior passed acceptance gates",
                )
            return

        if state.accepted_label is not None:
            incumbent = state.accepted_label
            agrees = read.resolved and read.label == incumbent

            if agrees:
                if state.phase == "challenged":
                    self._emit(
                        concept,
                        "challenge_cleared",
                        incumbent,
                        incumbent,
                        incumbent,
                        read,
                        "posterior returned to the accepted label",
                    )
                state.phase = "stable"
                state.candidate_label = None
                state.conflict_streak = 0
                state.candidate_streak = 0
                return

            challenger = (
                read.label
                if read.resolved and read.label != incumbent
                else (
                    posterior_candidate
                    if posterior_candidate != incumbent
                    else None
                )
            )
            state.conflict_streak += 1
            state.phase = "challenged"
            state.candidate_label = challenger

            if state.conflict_streak == 1:
                self._emit(
                    concept,
                    "challenged",
                    incumbent,
                    incumbent,
                    challenger,
                    read,
                    "current posterior no longer supports the incumbent label",
                )

            if state.conflict_streak >= self.challenge_patience:
                state.previous_label = incumbent
                state.accepted_label = None
                state.phase = "unresolved"
                state.candidate_label = (
                    read.label if read.resolved else challenger
                )
                state.candidate_streak = (
                    1 if read.resolved and read.label is not None else 0
                )
                self._emit(
                    concept,
                    "withdrawn",
                    incumbent,
                    None,
                    state.candidate_label,
                    read,
                    "sustained semantic conflict withdrew the incumbent",
                )
            return

        # The label has already been withdrawn. A replacement needs sustained
        # resolved posterior support.
        if read.resolved and read.label is not None:
            if state.candidate_label == read.label:
                state.candidate_streak += 1
            else:
                state.candidate_label = read.label
                state.candidate_streak = 1

            if state.candidate_streak >= self.revision_patience:
                old = state.previous_label
                new = read.label
                state.accepted_label = new
                state.previous_label = None
                state.phase = "stable"
                state.conflict_streak = 0
                state.candidate_streak = 0
                state.candidate_label = None
                self._emit(
                    concept,
                    "recovered" if old == new else "revised",
                    old,
                    new,
                    new,
                    read,
                    (
                        "previous label regained sustained support"
                        if old == new
                        else "challenger earned sustained replacement support"
                    ),
                )
            return

        state.phase = "unresolved"
        state.candidate_streak = 0
        state.candidate_label = (
            posterior_candidate
            if posterior_candidate != state.previous_label
            else None
        )

    def advance(
        self,
        epoch: int,
        *,
        neural: Iterable[NeuralLabelEvidence] = (),
        structural: Iterable[StructuralSemanticObservation] = (),
    ) -> tuple[RevisionRead, ...]:
        if epoch <= self.epoch:
            raise ValueError("epochs must advance strictly")

        neural_items = tuple(neural)
        structural_items = tuple(structural)

        for item in neural_items:
            self._neural_history.append(
                TimedNeuralEvidence(epoch=epoch, evidence=item)
            )
        for item in structural_items:
            self._structural_history.append(
                TimedStructuralEvidence(epoch=epoch, evidence=item)
            )

        self.epoch = int(epoch)
        self._prune()
        self._grounder = self._rebuild_grounder()

        grounding_reads = {
            concept: self._grounder.read(concept)
            for concept in self.concepts
        }
        for concept in self.concepts:
            self._advance_state(concept, grounding_reads[concept])

        self._reads = {}
        for concept in self.concepts:
            state = self._states[concept]
            grounding = grounding_reads[concept]
            self._reads[concept] = RevisionRead(
                epoch=self.epoch,
                concept=concept,
                phase=state.phase,
                accepted_label=state.accepted_label,
                candidate_label=state.candidate_label,
                conflict_streak=state.conflict_streak,
                candidate_streak=state.candidate_streak,
                posterior_label=grounding.label,
                posterior_confidence=grounding.confidence,
                posterior_margin=grounding.margin,
                neural_top_label=grounding.neural_top_label,
                neural_top_confidence=grounding.neural_top_confidence,
                grounding=grounding,
            )

        return tuple(self._reads[concept] for concept in self.concepts)

    def read(self, concept: Concept) -> RevisionRead:
        if concept not in self._states:
            raise ValueError("unknown concept")
        if concept not in self._reads:
            raise RuntimeError("advance() must be called before read()")
        return self._reads[concept]

    def accepted_mapping(self) -> dict[Concept, Label]:
        return {
            concept: state.accepted_label
            for concept, state in self._states.items()
            if state.accepted_label is not None
        }
