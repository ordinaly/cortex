"""Selective training with explicitly missing observations and fresh validation.

Stationary binary fixture only: decisions precede current answers/outcomes.
Counts are context/answer conditional; no unobserved joint law is imputed.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from math import sqrt
from random import Random
from types import MappingProxyType

from possibility_geometry import Candidate, FiniteModel, PossibilityFilter
from question_learning import QUESTIONS


@dataclass(frozen=True)
class PartialEvidence:
    episode_id: str
    context: int
    answers: tuple[int | None, int | None]
    outcome: int

    def __post_init__(self):
        object.__setattr__(self, 'answers', tuple(self.answers))
        if not self.episode_id or self.context not in (0, 1) or self.outcome not in (0, 1):
            raise ValueError('invalid evidence')
        if len(self.answers) != 2 or any(v not in (None, 0, 1) for v in self.answers):
            raise ValueError('binary or explicitly missing answers required')


class Counts:
    def __init__(self):
        self.root = {c: [0, 0] for c in (0, 1)}
        self.cells = {(c, q, v): [0, 0] for c in (0, 1) for q in QUESTIONS for v in (0, 1)}
        self.ids = set()

    def update(self, row):
        if row.episode_id in self.ids:
            raise ValueError('duplicate evidence')
        self.ids.add(row.episode_id)
        self.root[row.context][0] += row.outcome
        self.root[row.context][1] += 1
        for q, answer in zip(QUESTIONS, row.answers):
            if answer is not None:
                cell = self.cells[row.context, q, answer]
                cell[0] += row.outcome
                cell[1] += 1

    def snapshot(self):
        return Snapshot(MappingProxyType({k: tuple(v) for k, v in self.root.items()}),
                        MappingProxyType({k: tuple(v) for k, v in self.cells.items()}),
                        frozenset(self.ids))


@dataclass(frozen=True)
class Snapshot:
    root: object
    cells: object
    ids: frozenset[str]

    def p(self, context, question=None, answer=None):
        count = self.root[context] if question is None else self.cells[context, question, answer]
        return F(count[0]+1, count[1]+2)

    def n(self, context, question):
        return sum(self.cells[context, question, v][1] for v in (0, 1))

    def prevalence(self, context, question):
        return F(self.cells[context, question, 1][1]+1, self.n(context, question)+2)

    def gain(self, context, question):
        p = self.prevalence(context, question)
        return p * (1-p) * (self.p(context, question, 1)-self.p(context, question, 0))**2

    def propose(self):
        total = sum(v[1] for v in self.root.values())
        if not total:
            raise ValueError('no discovery evidence')
        return max(QUESTIONS, key=lambda q: (sum(self.root[c][1]*self.gain(c, q) for c in (0, 1))/total, q))

    def predict(self, context, selected=None, answer=None):
        """Exact two-alternative model only for the selected question.

        We do not manufacture a joint distribution of unobserved probe pairs.
        """
        if selected is None:
            return self.p(context)
        if selected not in QUESTIONS or answer not in (None, 0, 1):
            raise ValueError('invalid question or answer')
        prevalence = self.prevalence(context, selected)
        rows = {}
        for v in (0, 1):
            p = self.p(context, selected, v)
            rows[str(v), selected] = {(str(v), str(v)): 1}
            rows[str(v), 'outcome'] = {(str(v), '1'): p, (str(v), '0'): 1-p}
        model = FiniteModel('selective-training-v0.3', (Candidate('0'), Candidate('1')),
                            (selected, 'outcome'), ('0', '1'), rows)
        learner = PossibilityFilter(model, {'0': 1-prevalence, '1': prevalence})
        if answer is not None:
            learner.predict(selected)
            learner.assimilate(selected, str(answer))
        return learner.predict('outcome').outcomes['1']


class Acquisition:
    """Heuristic variance-separation score + count bonus, with epsilon exploration.

    Not a calibrated value-of-information computation or confidence bound.
    """
    def __init__(self, seed, epsilon=.2):
        if not 0 <= epsilon <= 1:
            raise ValueError('invalid exploration rate')
        self.rng = Random(seed)
        self.epsilon = epsilon

    def choose(self, snapshot, context):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(QUESTIONS), 'explore'
        scores = {q: float(snapshot.gain(context, q)) + .1/sqrt(snapshot.n(context, q)+1)
                  for q in QUESTIONS}
        best = max(scores.values())
        return self.rng.choice([q for q in QUESTIONS if scores[q] == best]), 'score'


@dataclass(frozen=True)
class Validation:
    proposed: str
    adopted: str | None
    gain: F
    ids: frozenset[str]


def validate(snapshot, proposal, records, margin=F(1, 50), min_cell=4):
    records = tuple(records)
    if proposal not in QUESTIONS or not records or margin < 0 or min_cell < 1:
        raise ValueError('invalid validation configuration')
    ids = [r.episode_id for r in records]
    if len(set(ids)) != len(ids) or set(ids) & snapshot.ids:
        raise ValueError('validation must use fresh evidence')
    i = QUESTIONS.index(proposal)
    if any(r.answers[i] is None for r in records):
        raise ValueError('validation must acquire the proposed question')
    gain = sum(((snapshot.p(r.context)-r.outcome)**2 -
                (snapshot.p(r.context, proposal, r.answers[i])-r.outcome)**2
                for r in records), F(0))/len(records)
    supported = all(snapshot.cells[c, proposal, v][1] >= min_cell for c in (0, 1) for v in (0, 1))
    return Validation(proposal, proposal if supported and gain > margin else None, gain, frozenset(ids))
