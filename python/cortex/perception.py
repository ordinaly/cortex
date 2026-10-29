"""Model-neutral perception boundary. Names never determine entity bindings.

Neural models implement PerceptionProvider; Cortex owns persistent identities.
Typed relation proposals remain outside the native binary relation channel until
explicitly accepted. Scores are provider scores, not calibrated truth values.
"""
from dataclasses import dataclass
from math import isfinite
from json import dumps
from typing import Protocol

from .autonomy import CortexStepRuntime
from .knowledge import KnowledgeStatement


def _score(value):
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError('score must be finite and in [0, 1]')


@dataclass(frozen=True)
class NameCandidate:
    text: str
    score: float
    language: str = 'en'
    kind: str = 'category'

    def __post_init__(self):
        if not self.text.strip() or not self.language or self.kind not in ('category', 'name', 'alias'):
            raise ValueError('invalid name candidate')
        _score(self.score)


@dataclass(frozen=True)
class Detection:
    detection_id: str
    features: tuple[float, ...]
    names: tuple[NameCandidate, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'features', tuple(self.features))
        object.__setattr__(self, 'names', tuple(self.names))
        if not self.detection_id or not self.features or not all(isfinite(x) for x in self.features):
            raise ValueError('detection needs identity and finite features')


@dataclass(frozen=True)
class RelationProposal:
    proposal_id: str
    subject: str
    relation: str
    object: str
    score: float

    def __post_init__(self):
        if not all((self.proposal_id, self.subject, self.relation, self.object)) or self.relation.startswith('@'):
            raise ValueError('invalid relation proposal')
        _score(self.score)


@dataclass(frozen=True)
class PerceptionFrame:
    frame_id: str
    timestamp: float
    source: str
    model: str
    feature_space: str
    detections: tuple[Detection, ...]
    relations: tuple[RelationProposal, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'detections', tuple(self.detections))
        object.__setattr__(self, 'relations', tuple(self.relations))
        if not all((self.frame_id, self.source, self.model, self.feature_space)) or not isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError('frame needs provenance and nonnegative finite timestamp')
        ids = {d.detection_id for d in self.detections}
        if len(ids) != len(self.detections) or len({len(d.features) for d in self.detections}) > 1:
            raise ValueError('duplicate detections or inconsistent dimensions')
        if len({r.proposal_id for r in self.relations}) != len(self.relations):
            raise ValueError('duplicate relation proposal')
        if any(r.subject not in ids or r.object not in ids for r in self.relations):
            raise ValueError('relation endpoint is not a detection in this frame')


class PerceptionProvider(Protocol):
    def perceive(self, image, *, frame_id: str, timestamp: float, source: str) -> PerceptionFrame: ...


@dataclass(frozen=True)
class NameEvidence:
    entity_id: int
    frame_id: str
    detection_id: str
    source: str
    timestamp: float
    model: str
    candidate: NameCandidate


@dataclass(frozen=True)
class BoundRelation:
    frame_id: str
    proposal: RelationProposal
    subject_id: int
    object_id: int
    source: str
    timestamp: float
    model: str


@dataclass(frozen=True)
class PerceptionRead:
    frame_id: str
    bindings: tuple[tuple[str, int], ...]
    relations: tuple[BoundRelation, ...]
    unresolved: bool


class PerceptionAdapter:
    """One chronological stream, one explicitly versioned feature coordinate space.

    Configure the runtime for this embedding dimension with identity-only nuisance
    transforms unless transformations are justified for that neural encoder.
    Native runtime mutation cannot be rolled back here; a failed runtime call or
    invalid return poisons the adapter so later calls cannot silently continue.
    """
    def __init__(self, runtime: CortexStepRuntime, *, feature_space: str, feature_dim: int,
                 max_frames=1000):
        if not feature_space or feature_dim < 1 or max_frames < 1:
            raise ValueError('feature contract and positive bounds required')
        self.runtime = runtime
        self.feature_space, self.feature_dim, self.max_frames = feature_space, feature_dim, max_frames
        self.names: list[NameEvidence] = []
        self.reads: dict[str, PerceptionRead] = {}
        self.last_timestamp = -1.0
        self.failed = False

    def observe(self, frame: PerceptionFrame):
        if self.failed:
            raise RuntimeError('adapter failed; restore runtime and create a new adapter')
        if frame.frame_id in self.reads or frame.timestamp <= self.last_timestamp:
            raise ValueError('frame ID must be fresh and timestamps strictly increasing')
        if len(self.reads) >= self.max_frames:
            raise RuntimeError('frame budget exceeded')
        if frame.feature_space != self.feature_space or any(len(d.features) != self.feature_dim for d in frame.detections):
            raise ValueError('feature space or dimension changed; explicit migration required')
        bindings, relations, evidence = (), (), []
        unresolved = False
        if frame.detections:
            try:
                # Typed directed proposals cannot be silently flattened into the
                # runtime's generic undirected binary relation observations.
                raw = self.runtime.step([list(d.features) for d in frame.detections], [], None, [], None)
                ids = tuple(raw.bindings)
                if (len(ids) != len(frame.detections) or len(set(ids)) != len(ids) or
                        any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids)):
                    raise ValueError('runtime returned invalid/noninjective bindings')
                unresolved = bool(getattr(raw, 'unresolved', False))
            except Exception:
                self.failed = True
                raise
            bindings = tuple((d.detection_id, i) for d, i in zip(frame.detections, ids))
            if not unresolved:
                mapping = dict(bindings)
                evidence = [NameEvidence(i, frame.frame_id, d.detection_id, frame.source,
                                         frame.timestamp, frame.model, n)
                            for d, i in zip(frame.detections, ids) for n in d.names]
                relations = tuple(BoundRelation(frame.frame_id, r, mapping[r.subject], mapping[r.object],
                                                 frame.source, frame.timestamp, frame.model)
                                  for r in frame.relations)
        read = PerceptionRead(frame.frame_id, bindings, relations, unresolved)
        self.names.extend(evidence)
        self.reads[frame.frame_id] = read
        self.last_timestamp = frame.timestamp
        return read

    def label(self, entity_id, *, language='en', minimum_score=.5, ambiguity_margin=.1):
        """Display decoration only; competing labels remain in the evidence log.

        Maximum observed score per text avoids multiplying correlated video-frame
        confidence. This is a simple display policy, not evidence calibration.
        """
        _score(minimum_score)
        _score(ambiguity_margin)
        scores = {}
        for e in self.names:
            n = e.candidate
            if e.entity_id == entity_id and n.language == language:
                scores[n.text] = max(scores.get(n.text, 0), n.score)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        fallback = f'entity:{entity_id}'
        if not ranked or ranked[0][1] < minimum_score:
            return fallback
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] <= ambiguity_margin:
            return fallback
        return f'{ranked[0][0]} [{fallback}]'

    def accepted_statements(self, frame_id, proposal_ids):
        """Explicit caller acceptance; never invoked automatically by observe."""
        read = self.reads[frame_id]
        proposals = {r.proposal.proposal_id: r for r in read.relations}
        requested = tuple(proposal_ids)
        if len(set(requested)) != len(requested) or any(i not in proposals for i in requested):
            raise ValueError('acceptance must name distinct bound proposals')
        return tuple(KnowledgeStatement(
            statement_id=dumps((proposals[i].source, frame_id, proposals[i].model, i)),
            subject=f'entity:{proposals[i].subject_id}',
            property=proposals[i].proposal.relation, value=f'entity:{proposals[i].object_id}',
            references=(proposals[i].source,), provenance=f'{proposals[i].model}; t={proposals[i].timestamp}')
            for i in requested)
