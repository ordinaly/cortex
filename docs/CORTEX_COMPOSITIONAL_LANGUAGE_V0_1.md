# Compositional language v0.1

Status: opt-in research layer, 21 October 2026. Native runtime unchanged.

## Purpose and existing foundations

Cortex already has specialized kinship composition, structural analogy,
algebraic-form induction, a relational hypergraph and a proof-dependency engine.
This milestone adds a reusable typed relational language above those foundations.
It reuses `research/proof_dependency.py` for derivation, alternate proofs,
contradiction reporting and fact retraction; it does not replace that engine.

Implementation: `research/compositional_language.py`.

Primitive symbol identities and role types are supplied by the caller. Invariant
recognition is an upstream responsibility: this layer does not establish that
an upstream basin is a correct or stable perceptual invariant.

## Language and composition

A `Symbol` has a name, version and ordered argument types. An `Atom` applies it
to typed entities or variables and has explicit positive or negative polarity.
The symbol and its version form its identity; conflicting signatures are rejected.
A three-place relation can preserve giver/object/recipient positions rather than
flattening the participants into undirected pairwise associations.

A `Schema` consists of conjunctive premises and a conclusion. Variables are
local to the schema. Every conclusion variable must occur in its premises.
Distinct-role constraints can require particular variables to denote different
entities. Grounding enforces types, equality, inequality, constants and argument
order. There are no function symbols or existentially created output entities.

Named motifs compose through ordinary schemas: a body can contain two motif
applications sharing a role, and the head names their composition. Grounding
makes variable scope explicit; unrelated local variable names do not capture one
another. This is conjunctive relational composition, not an assumption that all
relations are associative, symmetric or transitive.

## Abstraction from observations

`invent_motif` accepts at least two caller-selected examples of a small motif.
Exact canonicalization removes entity names and conjunction order while preserving
primitive symbol identity/version, argument roles, types and the equality pattern.
For at most four atoms it enumerates edge orders and assigns canonical variable
names by first occurrence. Matching shapes receive a deterministic generated
symbol. Every role is exposed; distinct entities in the examples remain distinct
variables with explicit inequality constraints when matched later.

This is bounded expression invention, not unrestricted language discovery.
Candidate subgraphs are supplied; graph-wide mining, sensor discovery, and learning
which roles to hide or identify are deferred. Motifs may be disconnected conjunctions.
The name is an abbreviation with a forward definition rule; reverse unfolding
is not implemented. Original evidence is retained, so no measured storage reduction
is claimed. Two examples establish recurrence in this fixture, not statistical
independence or generality.

## Empirical rules versus definitions

`propose_rule` abstracts shared premise/conclusion structure from positive,
caller-selected examples into a variable-bearing **hypothesis**. It does not
infer causality, a universal law, or a calibrated probability from repetition.
It preserves observed distinct-role constraints conservatively.

`validate_rule` tests that frozen hypothesis against fresh explicitly labeled
cases. A case is supporting or refuting only if the rule derives the queried
positive atom from its premises. Otherwise it is untested. Missing observations
are never interpreted as negation. The simple gate requires two supporting cases
and no explicit counterexample; it is not an anytime-valid statistical test.
Discovery IDs cannot be reused for validation, but distinct IDs alone do not prove
that evidence is independent. Validation targets and candidate motifs are supplied.

The reasoning engine can also explore an explicitly supplied hypothesis before
validation. Callers must distinguish such conditional exploration from adopted
knowledge. `Schema.kind` and `evidence_ids` retain that provenance; the demo admits
its empirical rule only after the fresh-validation gate passes.

## Context, proof and revision

Every fact carries a source ID and explicit context. Premises from different
contexts cannot join. Rules are currently universal across supplied contexts;
additional applicability conditions must be represented as premises. This does
not discover context boundaries or automatically interpret time.

Deduction uses the existing finite proof DAG. Explanations include parent proofs,
source facts and grounded rule dependencies. `LanguageModel.origins` maps each
grounded rule back to the abstract schema, context and variable binding. Alternative
proofs survive removal of one evidence source. Facts can be retracted/restored;
schemas can be disabled/re-enabled, with dependent conclusions rebuilt.

Positive and negative support can coexist. Queries distinguish supported, refuted,
conflict and unproven. A conflict does not imply arbitrary conclusions. Existing
positive support can still participate in explicitly licensed downstream rules;
this is support-based reasoning, not a consistency requirement on every premise.
No probability multiplication or proof-confidence calibration is performed.

`from_hyperedge` explicitly translates a caller-selected native
`RelationalHyperedge` using property/subject/object basin IDs. It preserves direction
and the source ID. Soft support is **not** interpreted as logical truth or proof
probability. The caller chooses the edge as an assumption and retains the original
weighted evidence. There is no automatic promotion of hypergraph edges to facts.

## Finite boundary

The model grounds variables over entities appearing in a context plus explicit
schema constants. By default at most 10,000 assignments may be examined and at most
256 derived proofs per literal are accepted by the existing engine. Exceeding a
budget raises an error rather than returning a partial result as a complete answer.
The grounded literal graph must be acyclic; recursive rules are rejected. Exact
grounding can be expensive. This prototype does not claim scalability or a memory
advantage over direct symbolic reasoning. Symbol-version migration is not automatic.

## Validation

```bash
PYTHONPATH=python python -m pytest tests/test_compositional_language.py tests/test_proof_dependency.py tests/test_kinship_composition.py tests/test_structural_analogy.py
python research/compositional_language_demo.py
```

The deterministic demo:

1. Invents one motif symbol from two renamed instances.
2. Proposes a relational implication and validates it on two fresh instances.
3. Infers a relation on unfamiliar entities from premises alone.
4. Shows the conclusion becoming unproven on evidence retraction, returning on
   restoration, and becoming unproven on rule withdrawal.
5. Rejects the same candidate when fresh validation includes a counterexample.

Tests also cover nested motif composition, role reversal, repeated/distinct entity
constraints, higher-arity typing, context isolation, explicit contradictions,
alternate proofs, unsafe heads, symbol versions, grounding/proof budgets and the
native hyperedge adapter. Unit coverage is included in ordinary PR CI. These are
correctness fixtures, not comparative evidence of autonomous reasoning.

## Next experiment

Let Cortex search a bounded region of its own relational evidence for candidate
motifs, then compare a new expression/rule against a direct-memory baseline on
prospective observations. Charge the vocabulary, retained witnesses and matching
costs. Add noisy and conflicting evidence before allowing automatic promotion
into the native controller. The question-learning layer can then seek observations
that distinguish competing rule scopes rather than only competing binary features.
