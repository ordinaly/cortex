"""Conservative online revision of a finite, single-stratum rule portfolio.

Predict before revealing labels. A counterexample quarantines only the rules
that predicted it. Replacement proposals and validation use successive disjoint
blocks. An observed failure is evidence of mismatch, not a diagnosis of drift.
"""
from copy import deepcopy
from dataclasses import dataclass, field

from compositional_language import Atom, LanguageModel
from pattern_discovery import Discovery, Episode, discover, select


@dataclass(frozen=True)
class Counterexample:
    episode_id: str
    rule_id: str
    predicted: Atom
    observed: Atom
    fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class RevisionStep:
    episode_id: str
    predictions: tuple[int, ...]
    correct: int
    wrong: int
    abstained: int
    witnesses: tuple[Counterexample, ...]
    admitted: tuple[str, ...]


@dataclass
class _Repair:
    proposal: list[Episode] = field(default_factory=list)
    validation: list[Episode] = field(default_factory=list)
    discovery: Discovery | None = None


class RevisingLanguage:
    """Bounded batch-in-stream repair; assumes reliable explicit labels.

    No inferred negative facts, fabricated hidden context or automatic causal
    diagnosis. Positive replacement rules only, matching the existing miner.
    A false alarm can suspend a useful rule until fresh evidence revalidates it.
    """
    def __init__(self, selection, *, block_size=4, max_steps=1000, max_examined=10000,
                 counterexample_limit=1, evidence_window=4):
        if (block_size < 2 or max_steps < 1 or max_examined < 1 or
                not 1 <= counterexample_limit <= evidence_window):
            raise ValueError('positive bounds and blocks of at least two required')
        heads = {r.head.symbol for r in selection.schemas}
        if any(r.kind != 'hypothesis' or r.head.polarity != 1 or
               any(a.symbol in heads for a in r.body) for r in selection.schemas):
            raise ValueError('positive single-stratum hypotheses required')
        self.active = {r.rule_id: r for r in selection.schemas}
        self.quarantined = {}
        self.pending = {}
        self.strikes = {}
        self.counterexample_limit = counterexample_limit
        self.evidence_window = evidence_window
        self.used_ids = set(selection.used_episode_ids)
        self.block_size, self.max_steps, self.max_examined = block_size, max_steps, max_examined
        self.steps = []
        self.searches = self.examined = self.validation_rounds = 0

    def step(self, episode):
        """Score pre-update predictions, then consume labels and revise atomically."""
        if episode.episode_id in self.used_ids:
            raise ValueError('episode must be fresh')
        if len(self.steps) >= self.max_steps:
            raise RuntimeError('stream budget exceeded')
        # Failures in bounded mining/grounding leave the original state intact.
        trial = deepcopy(self)
        result = trial._step(episode)
        self.__dict__.update(trial.__dict__)
        return result

    def _step(self, episode):
        context = episode.facts[0].context
        model = LanguageModel(episode.facts, tuple(self.active.values()))
        predictions = tuple(model.explain(Atom(a.symbol, a.terms), context).state
                            for a in episode.outcomes)
        witnesses = []
        failed = set()
        # Attribute using each rule's own proof, not merely a shared head symbol.
        for rule in tuple(self.active.values()):
            individual = LanguageModel(episode.facts, [rule])
            supported = contradicted = False
            for outcome in episode.outcomes:
                if outcome.symbol != rule.head.symbol:
                    continue
                query = Atom(outcome.symbol, outcome.terms)
                explanation = individual.explain(query, context)
                if explanation.state != 1:
                    continue
                if outcome.polarity == 1:
                    supported = True
                    continue
                contradicted = True
                node = next(n for n in explanation.nodes
                            if n.proof_id == explanation.selected_positive_id)
                witnesses.append(Counterexample(episode.episode_id, rule.rule_id, query,
                                               outcome, node.fact_dependencies))
            # At most one strike per applicable episode. A mixed episode counts
            # as a counterexample; missing labels do not expire evidence.
            if contradicted or supported:
                recent = self.strikes.get(rule.rule_id, []) + [contradicted]
                self.strikes[rule.rule_id] = recent[-self.evidence_window:]
                if sum(self.strikes[rule.rule_id]) >= self.counterexample_limit:
                    failed.add(rule.rule_id)
        for rule_id in failed:
            rule = self.active.pop(rule_id)
            self.quarantined[rule_id] = rule
            self.pending.setdefault(rule.head.symbol, _Repair())

        admitted = []
        for symbol, repair in tuple(self.pending.items()):
            outcomes = tuple(a for a in episode.outcomes if a.symbol == symbol)
            if not outcomes:
                continue  # Missing labels cannot validate or refute a candidate.
            sample = Episode(episode.episode_id, episode.facts, outcomes)
            if repair.discovery is None:
                repair.proposal.append(sample)
                if len(repair.proposal) == self.block_size:
                    repair.discovery = discover(repair.proposal, max_examined=self.max_examined)
                    self.searches += 1
                    self.examined += repair.discovery.examined
            else:
                repair.validation.append(sample)
                if len(repair.validation) == self.block_size:
                    chosen = select(repair.discovery, repair.validation)
                    self.validation_rounds += 1
                    # Keep the whole portfolio in one stratum, including retained rules.
                    combined = [*self.active.values(), *chosen.schemas]
                    heads = {r.head.symbol for r in combined}
                    safe = not any(a.symbol in heads for r in combined for a in r.body)
                    if safe and chosen.schemas:
                        for rule in chosen.schemas:
                            if rule.rule_id not in self.active:
                                admitted.append(rule.rule_id)
                            self.active[rule.rule_id] = rule
                            self.strikes[rule.rule_id] = []
                            self.quarantined.pop(rule.rule_id, None)
                        del self.pending[symbol]
                    else:
                        self.pending[symbol] = _Repair()
        correct = sum(p == a.polarity for p, a in zip(predictions, episode.outcomes))
        wrong = sum(p != 0 and p != a.polarity for p, a in zip(predictions, episode.outcomes))
        result = RevisionStep(episode.episode_id, predictions, correct, wrong,
                              predictions.count(0), tuple(witnesses), tuple(sorted(admitted)))
        self.used_ids.add(episode.episode_id)
        self.steps.append(result)
        return result
