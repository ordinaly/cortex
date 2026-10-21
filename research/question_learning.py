"""Bounded question learning over the exact possibility oracle.

A supplied binary probe vocabulary is explored, one conditional distinction is
proposed, and fresh evidence must validate it. This is feature selection, not
invention of sensors, causal discovery, or native Cortex runtime integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from types import MappingProxyType

from possibility_geometry import Candidate, FiniteModel, PossibilityFilter

QUESTIONS = ('probe_0', 'probe_1')
VALUES = tuple(product((0, 1), repeat=2))


@dataclass(frozen=True)
class Evidence:
    episode_id: str
    context: int
    answers: tuple[int, int]
    outcome: int

    def __post_init__(self):
        object.__setattr__(self, 'answers', tuple(self.answers))
        if not self.episode_id or self.context not in (0, 1) or self.outcome not in (0, 1):
            raise ValueError('invalid evidence')
        if len(self.answers) != 2 or any(x not in (0, 1) for x in self.answers):
            raise ValueError('two binary probe answers required')


@dataclass(frozen=True)
class SplitDecision:
    proposed: str
    adopted: str | None
    discovery_gain: F
    validation_gain: F
    margin: F
    discovery_count: int
    validation_count: int
    status: str


@dataclass(frozen=True)
class LearnedModel:
    """Beta(1,1) outcome means and Dirichlet(1,1,1,1) joint feature means.

    Counts expose evidence strength; predictive means do not represent a full
    posterior over model parameters. No calibrated parameter intervals claimed.
    """
    contexts: tuple[int, ...]
    root_counts: object
    split_counts: object
    joint_counts: object
    evidence_ids: frozenset[str]
    decision: SplitDecision

    def outcome_probability(self, context, feature=None, value=None):
        if context not in self.contexts:
            raise ValueError('unseen context requires external revision')
        if feature is None:
            ones, total = self.root_counts[context]
        else:
            if feature not in QUESTIONS or value not in (0, 1):
                raise ValueError('invalid conditional question')
            ones, total = self.split_counts[context, feature, value]
        return F(ones + 1, total + 2)

    def finite_model(self, context):
        if context not in self.contexts:
            raise ValueError('unseen context requires external revision')
        selected = self.decision.adopted
        candidates = tuple(Candidate(f'{a}{b}', (f'context:{context}',),
                                     (f'{QUESTIONS[0]}={a}', f'{QUESTIONS[1]}={b}')) for a, b in VALUES)
        rows, prior = {}, {}
        total = sum(self.joint_counts[context, pair] for pair in VALUES)
        for pair, candidate in zip(VALUES, candidates):
            prior[candidate.name] = F(self.joint_counts[context, pair] + 1, total + 4)
            for index, question in enumerate(QUESTIONS):
                rows[candidate.name, question] = {(candidate.name, str(pair[index])): F(1)}
            p = self.outcome_probability(context, selected,
                                         pair[QUESTIONS.index(selected)] if selected else None)
            rows[candidate.name, 'outcome'] = {(candidate.name, '1'): p, (candidate.name, '0'): 1-p}
        model = FiniteModel(f'question-v0.2:{context}:{selected}', candidates,
                            QUESTIONS + ('outcome',), ('0', '1'), rows)
        return model, MappingProxyType(prior)

    def probability(self, context, question=None, answer=None):
        """Use the exact reference for both question assimilation and forecasting."""
        model, prior = self.finite_model(context)
        learner = PossibilityFilter(model, prior)
        if question is not None:
            if question not in QUESTIONS or answer not in (0, 1):
                raise ValueError('invalid answer')
            learner.predict(question)
            learner.assimilate(question, str(answer))
        return learner.predict('outcome').outcomes['1']

    def question_values(self, context):
        """Expected reduction in binary-decision error, before acquisition cost."""
        model, prior = self.finite_model(context)
        baseline = self.probability(context)
        initial_risk = min(baseline, 1-baseline)
        values = {}
        for question in QUESTIONS:
            forecast = PossibilityFilter(model, prior).predict(question)
            risk = F(0)
            for answer in (0, 1):
                p = self.probability(context, question, answer)
                risk += forecast.outcomes[str(answer)] * min(p, 1-p)
            values[question] = initial_risk - risk
        return MappingProxyType(values)

    def choose_question(self, context, cost):
        cost = F(cost)
        if cost < 0:
            raise ValueError('cost must be nonnegative')
        values = self.question_values(context)
        best = max(QUESTIONS, key=lambda q: (values[q], q))
        return best if values[best] > cost else None


def fit(discovery, validation, *, margin=F(1, 50), min_cell=16):
    """Freeze the proposal on discovery data, validate once, never refit on test.

    The margin is an empirical benchmark gate, not a confidence bound. Evidence
    IDs must be distinct across and within phases; IDs alone do not establish
    statistical independence. All probe acquisition costs
    belong to the driver; these records contain only actually obtained answers.
    """
    discovery, validation = tuple(discovery), tuple(validation)
    margin = F(margin)
    if not discovery or not validation or margin < 0 or min_cell < 1:
        raise ValueError('nonempty phases and valid gate parameters required')
    ids = [r.episode_id for r in discovery + validation]
    if len(set(ids)) != len(ids):
        raise ValueError('discovery and validation evidence must be independent records')
    contexts = tuple(sorted({r.context for r in discovery}))
    if any(r.context not in contexts for r in validation):
        raise ValueError('validation includes unseen context')
    root = {c: [0, 0] for c in contexts}
    split = {(c, q, v): [0, 0] for c in contexts for q in QUESTIONS for v in (0, 1)}
    joint = {(c, pair): 0 for c in contexts for pair in VALUES}
    for r in discovery:
        root[r.context][0] += r.outcome
        root[r.context][1] += 1
        joint[r.context, r.answers] += 1
        for i, q in enumerate(QUESTIONS):
            split[r.context, q, r.answers[i]][0] += r.outcome
            split[r.context, q, r.answers[i]][1] += 1
    def probability(count):
        ones, total = count
        return F(ones+1, total+2)
    def gain(rows, q):
        i = QUESTIONS.index(q)
        return sum(((probability(root[r.context])-r.outcome)**2 -
                    (probability(split[r.context, q, r.answers[i]])-r.outcome)**2
                    for r in rows), F(0)) / len(rows)
    proposal = max(QUESTIONS, key=lambda q: (gain(discovery, q), q))
    discovery_gain, validation_gain = gain(discovery, proposal), gain(validation, proposal)
    supported = all(split[c, proposal, v][1] >= min_cell for c in contexts for v in (0, 1))
    adopt = supported and discovery_gain > margin and validation_gain > margin
    decision = SplitDecision(proposal, proposal if adopt else None, discovery_gain,
                             validation_gain, margin, len(discovery), len(validation),
                             'adopted' if adopt else 'insufficient_support' if not supported else 'rejected')
    return LearnedModel(contexts, MappingProxyType({k: tuple(v) for k, v in root.items()}),
                        MappingProxyType({k: tuple(v) for k, v in split.items()}),
                        MappingProxyType(joint), frozenset(ids), decision)
