"""Typed relational composition over Cortex's existing proof dependency engine.

Primitive symbols are caller-owned invariant identities. Motifs abstract entity
names and edge order, preserving argument roles, types and equality structure.
Rules are finite, range-restricted, function-free and acyclic after grounding.
Definition compression and empirical implication are explicitly different kinds.
"""
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, permutations, product
from json import dumps
from types import MappingProxyType

from proof_dependency import Literal, Rule as GroundRule, ProofDependencyModel


@dataclass(frozen=True, order=True)
class Symbol:
    name: str
    version: int
    roles: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, 'roles', tuple(self.roles))
        if not self.name or self.version < 1 or not self.roles or any(not r for r in self.roles):
            raise ValueError('symbol needs name, positive version and nonempty role types')


@dataclass(frozen=True, order=True)
class Entity:
    name: str
    kind: str = 'entity'


@dataclass(frozen=True, order=True)
class Variable:
    name: str
    kind: str = 'entity'


@dataclass(frozen=True)
class Atom:
    symbol: Symbol
    terms: tuple[Entity | Variable, ...]
    polarity: int = 1

    def __post_init__(self):
        object.__setattr__(self, 'terms', tuple(self.terms))
        if self.polarity not in (-1, 1) or len(self.terms) != len(self.symbol.roles):
            raise ValueError('invalid polarity or arity')
        if any(not isinstance(t, (Entity, Variable)) or not t.name or not t.kind or t.kind != role
               for t, role in zip(self.terms, self.symbol.roles)):
            raise ValueError('invalid term or role type')

    def bind(self, substitution):
        return Atom(self.symbol, tuple(substitution[t] if isinstance(t, Variable) else t for t in self.terms), self.polarity)

    def literal(self, context):
        if any(isinstance(t, Variable) for t in self.terms):
            raise ValueError('ground atom required')
        payload = (context, self.symbol.name, self.symbol.version, self.symbol.roles,
                   tuple((t.name, t.kind) for t in self.terms))
        return Literal(dumps(payload, separators=(',', ':')), self.polarity)


@dataclass(frozen=True)
class Schema:
    rule_id: str
    body: tuple[Atom, ...]
    head: Atom
    kind: str = 'hypothesis'
    evidence_ids: tuple[str, ...] = ()
    distinct: tuple[tuple[Variable, Variable], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'body', tuple(self.body))
        object.__setattr__(self, 'evidence_ids', tuple(self.evidence_ids))
        object.__setattr__(self, 'distinct', tuple(tuple(pair) for pair in self.distinct))
        if not self.rule_id or not self.body or self.kind not in ('definition', 'hypothesis'):
            raise ValueError('invalid rule schema')
        body_vars = {t for a in self.body for t in a.terms if isinstance(t, Variable)}
        if any(isinstance(t, Variable) and t not in body_vars for t in self.head.terms):
            raise ValueError('head variable must occur in premises')
        variables = body_vars | {t for t in self.head.terms if isinstance(t, Variable)}
        if any(len(pair) != 2 or pair[0] == pair[1] or any(v not in body_vars for v in pair) for pair in self.distinct):
            raise ValueError('invalid distinct-role constraint')
        if len({v.name for v in variables}) != len(variables):
            raise ValueError('one variable name cannot have different types')


@dataclass(frozen=True)
class Fact:
    fact_id: str
    context: str
    atom: Atom

    def __post_init__(self):
        if not self.fact_id or not self.context:
            raise ValueError('fact needs source ID and context')
        self.atom.literal(self.context)


@dataclass(frozen=True)
class MotifExample:
    evidence_id: str
    atoms: tuple[Atom, ...]

    def __post_init__(self):
        object.__setattr__(self, 'atoms', tuple(self.atoms))
        if not self.evidence_id or not self.atoms:
            raise ValueError('nonempty example required')
        for atom in self.atoms:
            atom.literal('example')


def canonical_motif(atoms, *, max_atoms=4):
    """Exact small-motif canonicalization; entity renaming and conjunction order."""
    atoms = tuple(atoms)
    if not atoms or len(atoms) > max_atoms:
        raise ValueError('motif atom budget exceeded or empty motif')
    if len(set(atoms)) != len(atoms):
        raise ValueError('duplicate motif atoms')
    best = None
    for order in permutations(atoms):
        mapping, body = {}, []
        for atom in order:
            atom.literal('motif')
            for term in atom.terms:
                if term not in mapping:
                    mapping[term] = Variable(f'v{len(mapping)}', term.kind)
            body.append(Atom(atom.symbol, tuple(mapping[t] for t in atom.terms), atom.polarity))
        key = dumps([(a.symbol.name, a.symbol.version, a.symbol.roles, a.polarity,
                      [(t.name, t.kind) for t in a.terms]) for a in body], separators=(',', ':'))
        if best is None or key < best[0]:
            best = key, tuple(body), tuple(mapping.values())
    return best


def invent_motif(examples):
    """Name a shared exact shape; does not infer that it causes another event.

    Caller supplies bounded candidate motifs. Graph-wide mining and role/type
    discovery are deferred. Every role is exposed to preserve all bindings.
    """
    examples = tuple(examples)
    if len(examples) < 2 or len({e.evidence_id for e in examples}) != len(examples):
        raise ValueError('at least two distinct evidence IDs required')
    canonical = [canonical_motif(e.atoms) for e in examples]
    if any(row[0] != canonical[0][0] for row in canonical):
        raise ValueError('examples do not share one role/equality structure')
    key, body, variables = canonical[0]
    symbol = Symbol('M_' + sha256(key.encode()).hexdigest()[:20], 1, tuple(v.kind for v in variables))
    return Schema('define:' + symbol.name, body, Atom(symbol, variables), 'definition',
                  tuple(e.evidence_id for e in examples), tuple(combinations(variables, 2)))


class LanguageModel:
    """Ground typed schemas and reuse the existing acyclic proof/dependency DAG.

    All deductions are conditional on supplied facts and active hypotheses.
    Contexts never join. Variable names are local to a schema. Retraction rebuilds
    from active evidence so alternate proofs survive when available.
    """
    def __init__(self, facts, schemas, *, max_groundings=10000, max_proofs=256):
        self.facts, self.schemas = tuple(facts), tuple(schemas)
        self.max_groundings, self.max_proofs = max_groundings, max_proofs
        if not self.facts or max_groundings < 1 or max_proofs < 1:
            raise ValueError('nonempty facts and positive budgets required')
        if len({f.fact_id for f in self.facts}) != len(self.facts):
            raise ValueError('duplicate fact ID')
        if len({r.rule_id for r in self.schemas}) != len(self.schemas):
            raise ValueError('duplicate schema ID')
        signatures = {}
        for atom in [f.atom for f in self.facts] + [a for r in self.schemas for a in (*r.body, r.head)]:
            key = (atom.symbol.name, atom.symbol.version)
            if key in signatures and signatures[key] != atom.symbol.roles:
                raise ValueError('symbol identity reused with incompatible signature')
            signatures[key] = atom.symbol.roles
        self._inactive_facts, self._inactive_rules = set(), set()
        self._rebuild()

    def _rebuild(self):
        rules, origins = [], {}
        attempts = 0
        contexts = sorted({f.context for f in self.facts})
        for context in contexts:
            # Domains include retracted entities so restoration has stable meaning.
            entities = {t for f in self.facts if f.context == context for t in f.atom.terms}
            for schema in self.schemas:
                if schema.rule_id in self._inactive_rules:
                    continue
                variables = sorted({t for a in schema.body for t in a.terms if isinstance(t, Variable)})
                constants = {t for a in (*schema.body, schema.head) for t in a.terms if isinstance(t, Entity)}
                domains = [sorted(e for e in entities | constants if e.kind == v.kind) for v in variables]
                for values in product(*domains):
                    attempts += 1
                    if attempts > self.max_groundings:
                        raise RuntimeError('grounding budget exceeded; no partial answer')
                    binding = dict(zip(variables, values))
                    if any(binding[a] == binding[b] for a, b in schema.distinct):
                        continue
                    payload = (schema.rule_id, context, [(v.name, e.name, e.kind) for v, e in zip(variables, values)])
                    rule_id = dumps(payload, separators=(',', ':'))
                    rules.append(GroundRule(rule_id, tuple(a.bind(binding).literal(context) for a in schema.body),
                                            schema.head.bind(binding).literal(context)))
                    origins[rule_id] = (schema.rule_id, context, tuple(binding.items()))
        facts = {f.fact_id: f.atom.literal(f.context) for f in self.facts}
        engine = ProofDependencyModel(facts, rules, max_proofs_per_literal=self.max_proofs)
        for fact_id in self._inactive_facts:
            engine.retract_fact(fact_id)
        engine.proof_nodes()  # Force cycle/proof-budget errors before exposing state.
        self._engine, self.origins = engine, MappingProxyType(origins)

    def explain(self, atom, context):
        if atom.polarity != 1:
            raise ValueError('query positive atom; explanation contains both polarities')
        return self._engine.explain(atom.literal(context).atom)

    def retract_fact(self, fact_id):
        self._engine.retract_fact(fact_id)
        self._inactive_facts.add(fact_id)

    def restore_fact(self, fact_id):
        self._engine.restore_fact(fact_id)
        self._inactive_facts.discard(fact_id)

    def set_rule_active(self, rule_id, active):
        if rule_id not in {r.rule_id for r in self.schemas}:
            raise ValueError('unknown schema')
        before = set(self._inactive_rules)
        if active:
            self._inactive_rules.discard(rule_id)
        else:
            self._inactive_rules.add(rule_id)
        try:
            self._rebuild()
        except Exception:
            self._inactive_rules = before
            raise


@dataclass(frozen=True)
class RuleExample:
    evidence_id: str
    premises: tuple[Atom, ...]
    observed: Atom

    def __post_init__(self):
        object.__setattr__(self, 'premises', tuple(self.premises))
        if not self.evidence_id or not self.premises:
            raise ValueError('nonempty rule example required')
        for atom in (*self.premises, self.observed):
            atom.literal('example')


def propose_rule(examples):
    """Abstract a shared implication from bounded, caller-selected positive cases.

    This proposes an empirical hypothesis, not a logical theorem or causal law.
    """
    examples = tuple(examples)
    if len(examples) < 2 or len({e.evidence_id for e in examples}) != len(examples):
        raise ValueError('at least two distinct proposal examples required')
    proposals = []
    for example in examples:
        if example.observed.polarity != 1 or len(example.premises) > 4:
            raise ValueError('positive proposal and bounded premises required')
        best = None
        for order in permutations(example.premises):
            mapping, body = {}, []
            for atom in order:
                for term in atom.terms:
                    if term not in mapping:
                        mapping[term] = Variable(f'v{len(mapping)}', term.kind)
                body.append(Atom(atom.symbol, tuple(mapping[t] for t in atom.terms), atom.polarity))
            if any(t not in mapping for t in example.observed.terms):
                raise ValueError('conclusion introduces unbound entity')
            head = Atom(example.observed.symbol, tuple(mapping[t] for t in example.observed.terms))
            key = dumps([(a.symbol.name, a.symbol.version, a.symbol.roles, a.polarity,
                          [(t.name, t.kind) for t in a.terms]) for a in (*body, head)], separators=(',', ':'))
            if best is None or key < best[0]:
                best = key, tuple(body), head
        proposals.append(best)
    if any(p[0] != proposals[0][0] for p in proposals):
        raise ValueError('examples disagree on role/equality structure')
    key, body, head = proposals[0]
    return Schema('H_' + sha256(key.encode()).hexdigest()[:20], body, head, 'hypothesis',
                  tuple(e.evidence_id for e in examples),
                  tuple(combinations(sorted({t for a in body for t in a.terms}), 2)))


@dataclass(frozen=True)
class RuleValidation:
    supported: int
    refuted: int
    untested: int
    accepted: bool


def validate_rule(schema, examples, *, minimum_support=2):
    """One-shot prospective gate over explicit labeled outcomes; absence is unknown."""
    examples = tuple(examples)
    ids = [e.evidence_id for e in examples]
    if minimum_support < 1 or len(ids) != len(set(ids)) or set(ids) & set(schema.evidence_ids):
        raise ValueError('fresh validation and positive support threshold required')
    supported = refuted = untested = 0
    for example in examples:
        model = LanguageModel([Fact(str(i), 'validation', a) for i, a in enumerate(example.premises)], [schema])
        query = Atom(example.observed.symbol, example.observed.terms)
        if model.explain(query, 'validation').state != 1:
            untested += 1
        elif example.observed.polarity == 1:
            supported += 1
        else:
            refuted += 1
    return RuleValidation(supported, refuted, untested, supported >= minimum_support and not refuted)


def from_hyperedge(edge, context, *, symbol_version=1):
    """Explicit hypothesis adapter for a Cortex RelationalHyperedge.

    Native numeric basin IDs supply entity/property identities. Soft support is
    NOT converted to logical truth or multiplied into proof confidence. The caller
    must choose which edges to assume; retain the original evidence for weighting.
    """
    symbol = Symbol(f'basin:{edge.property_id}', symbol_version, ('entity', 'entity'))
    return Fact(edge.edge_id, context, Atom(symbol, (Entity(f'basin:{edge.subject_id}'),
                                                    Entity(f'basin:{edge.object_id}'))))
