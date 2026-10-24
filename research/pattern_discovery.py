"""Bounded context-local motif mining and prospective rule selection.

Inputs are explicit assumed facts and separately observed outcomes. Entity and
symbol discovery remain upstream. Scores are declared decision costs, not bits
of compression or calibrated probabilities. No held-out labels enter mining.
"""
from dataclasses import dataclass
from itertools import combinations

from compositional_language import (
    Atom, Fact, LanguageModel, MotifExample, RuleExample, Schema,
    canonical_implication, canonical_motif, invent_motif, propose_rule,
)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    facts: tuple[Fact, ...]
    outcomes: tuple[Atom, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'facts', tuple(self.facts))
        object.__setattr__(self, 'outcomes', tuple(self.outcomes))
        if not self.episode_id or not self.facts:
            raise ValueError('episode needs identity and facts')
        if len({f.context for f in self.facts}) != 1:
            raise ValueError('episode must have one context')
        if len({f.fact_id for f in self.facts}) != len(self.facts):
            raise ValueError('duplicate fact ID')
        known = {(f.atom.symbol, f.atom.terms) for f in self.facts}
        labels = set()
        for atom in self.outcomes:
            atom.literal('outcome')
            key = atom.symbol, atom.terms
            if key in labels or key in known:
                raise ValueError('outcomes must be unique and withheld from premises')
            labels.add(key)
        # Validate symbol identity/signature consistency, including withheld data.
        signatures = {}
        for atom in [f.atom for f in self.facts] + list(self.outcomes):
            key = atom.symbol.name, atom.symbol.version
            if key in signatures and signatures[key] != atom.symbol.roles:
                raise ValueError('incompatible symbol signature')
            signatures[key] = atom.symbol.roles


@dataclass(frozen=True)
class Discovery:
    definitions: tuple[Schema, ...]
    hypotheses: tuple[Schema, ...]
    episode_ids: tuple[str, ...]
    examined: int


def _episodes(episodes):
    episodes = tuple(episodes)
    if len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError('duplicate episode ID')
    return episodes


def _connected(atoms):
    remaining = [set(a.terms) for a in atoms]
    reached = remaining.pop()
    while remaining:
        touching = [r for r in remaining if reached & r]
        if not touching:
            return False
        for terms in touching:
            reached.update(terms)
            remaining.remove(terms)
    return True


def discover(episodes, *, max_atoms=3, max_examined=10000, minimum_episodes=2):
    """Enumerate connected conjunctions; support counts distinct episodes.

    Bounds count rejected/disconnected subsets and candidate outcome pairings,
    not merely accepted motifs. Exceeding a bound fails without partial output.
    Candidate order and identities do not depend on input traversal order.
    """
    episodes = _episodes(episodes)
    if not 2 <= max_atoms <= 4 or max_examined < 1 or minimum_episodes < 2:
        raise ValueError('invalid discovery bounds')
    motifs, rules = {}, {}
    examined = 0

    def tick():
        nonlocal examined
        examined += 1
        if examined > max_examined:
            raise RuntimeError('discovery budget exceeded; no partial output')

    for episode in sorted(episodes, key=lambda e: e.episode_id):
        atoms = sorted({f.atom for f in episode.facts}, key=lambda a: a.literal('').atom + str(a.polarity))
        for size in range(2, min(max_atoms, len(atoms)) + 1):
            for subset in combinations(atoms, size):
                tick()
                if not _connected(subset):
                    continue
                key = canonical_motif(subset)[0]
                motifs.setdefault(key, {}).setdefault(episode.episode_id,
                    MotifExample(episode.episode_id, subset))
                entities = {t for a in subset for t in a.terms}
                for outcome in sorted(episode.outcomes, key=lambda a: a.literal('').atom):
                    tick()
                    if outcome.polarity != 1 or not set(outcome.terms) <= entities:
                        continue
                    # Conservative nonrecursive stratum: target symbol absent from body.
                    if any(a.symbol == outcome.symbol for a in subset):
                        continue
                    example = RuleExample(episode.episode_id, subset, outcome)
                    key = canonical_implication(example)[0]
                    rules.setdefault(key, {}).setdefault(episode.episode_id, example)
    definitions = tuple(invent_motif(tuple(rows.values())) for _, rows in sorted(motifs.items())
                        if len(rows) >= minimum_episodes)
    hypotheses = tuple(propose_rule(tuple(rows.values())) for _, rows in sorted(rules.items())
                       if len(rows) >= minimum_episodes)
    return Discovery(definitions, hypotheses, tuple(e.episode_id for e in episodes), examined)


@dataclass(frozen=True)
class Score:
    correct: int
    wrong: int
    abstained: int
    baseline_loss: float
    loss: float


def evaluate(schemas, episodes, *, error_cost=2.0, abstention_cost=1.0):
    """Predict explicit labeled queries only; an unobserved relation is unknown."""
    if not 0 < abstention_cost < error_cost < float('inf'):
        raise ValueError('finite error cost must exceed positive abstention cost')
    schemas = tuple(schemas)
    correct = wrong = abstained = 0
    for episode in _episodes(episodes):
        model = LanguageModel(episode.facts, schemas)
        context = episode.facts[0].context
        for outcome in episode.outcomes:
            state = model.explain(Atom(outcome.symbol, outcome.terms), context).state
            if state == 0:
                abstained += 1
            elif state == outcome.polarity:
                correct += 1
            else:
                wrong += 1
    return Score(correct, wrong, abstained, (correct + wrong + abstained) * abstention_cost,
                 wrong * error_cost + abstained * abstention_cost)


def complexity(schema):
    """Declared syntax units: atom occurrences plus argument occurrences."""
    return sum(1 + len(a.terms) for a in (*schema.body, schema.head))


@dataclass(frozen=True)
class Selection:
    schemas: tuple[Schema, ...]
    score: Score
    complexity_cost: float
    rejected: tuple[str, ...]
    used_episode_ids: tuple[str, ...]

    def test(self, episodes):
        episodes = _episodes(episodes)
        if set(self.used_episode_ids) & {e.episode_id for e in episodes}:
            raise ValueError('test episodes must be fresh')
        return evaluate(self.schemas, episodes)


def select(discovery, validation, *, cost_per_unit=0.1, minimum_correct=2):
    """Deterministic greedy portfolio selection using fresh validation only.

    Require zero explicit validation errors per candidate and marginal loss
    reduction greater than syntax cost. Redundant rules earn no marginal gain.
    This gate is heuristic selection, not a generalization guarantee.
    """
    validation = _episodes(validation)
    if set(discovery.episode_ids) & {e.episode_id for e in validation}:
        raise ValueError('validation episodes must be fresh')
    if not 0 <= cost_per_unit < float('inf') or minimum_correct < 1:
        raise ValueError('invalid selection costs')
    accepted, rejected, candidates = [], [], []
    # Only use one predictive stratum: a predicted symbol cannot be a premise.
    # This avoids accidental cross-rule cycles and unsupported multi-step claims.
    heads = {r.head.symbol for r in discovery.hypotheses}
    for rule in discovery.hypotheses:
        if any(a.symbol in heads for a in rule.body):
            rejected.append(rule.rule_id)
            continue
        score = evaluate([rule], validation)
        if score.wrong or score.correct < minimum_correct:
            rejected.append(rule.rule_id)
        else:
            candidates.append(rule)
    current = evaluate([], validation)
    cost = 0.0
    while candidates:
        options = []
        for rule in candidates:
            score = evaluate([*accepted, rule], validation)
            charge = cost_per_unit * complexity(rule)
            gain = current.loss - score.loss - charge
            options.append((gain, rule.rule_id, rule, score, charge))
        gain, _, rule, score, charge = sorted(options, key=lambda x: (-x[0], x[1]))[0]
        if gain <= 0:
            rejected.extend(r.rule_id for r in candidates)
            break
        accepted.append(rule)
        candidates.remove(rule)
        current = score
        cost += charge
    return Selection(tuple(accepted), current, cost, tuple(sorted(rejected)),
                     (*discovery.episode_ids, *(e.episode_id for e in validation)))
