# Cortex Reasoning Map v1

Status: **active research program**.

The Reasoning Map is a capability-oriented benchmark program for Cortex.

Its goal is not to produce one aggregate "intelligence score." Instead it asks
which distinct reasoning operations Cortex can perform, how they degrade, and
where its explicit unresolved state provides useful epistemic restraint.

## Capability map

| capability | benchmark | status | strongest current result |
|---|---|---|---|
| recursive typed composition | `kinship-composition-v1` | passed | 1.000 accuracy / 1.000 coverage on unseen depth 6–10 |
| candidate-law induction | `algebra-induction-v1` | passed | exact held-out recovery when supported laws were evidenced |
| bounded law-form discovery | `algebra-form-induction-v1.7` | passed | 375-form grammar; zero wrong resolved predictions in 120 cases |
| cross-world abstract transfer | `algebra-transfer-v1.6` | passed | +0.320 coverage over scratch at 20% target evidence; safe negative transfer |
| structural counterfactual reasoning | `counterfactual-reasoning-v1` | passed | all frozen gates passed across 140 worlds |
| belief revision under contradiction | `belief-revision-v1` | passed | retraction 7/14/21 batches for prior strengths 1/2/3; exact locality |
| structural analogy / isomorphism | `structural-analogy-v1` | passed | 0.9718 coverage with 12/90 facts; exact ambiguity preservation |
| latent concept induction | `latent-concept-induction-v1` | passed | exact recovery, ambiguity preservation, predictive gain, reversible lifecycle |
| explicit proof dependency | `explicit-proof-dependency-v1` | passed | all frozen gates passed across 120 cases; exact alternate-support handling |
| planning / replanning | planned | later | — |
| open-world novelty attribution | planned | later | — |

All passed entries are research fixtures, not stable runtime semantics unless
separately promoted through the executable-specification process.

## Why these axes are separated

A system that performs well on algebraic closure may still fail at causal
revision, analogy, planning, or concept invention.

The Reasoning Map therefore treats the following as distinct operations:

$$
\text{deduction},
\quad
\text{induction},
\quad
\text{transfer},
\quad
\text{counterfactual modification},
\quad
\text{revision},
\quad
\text{analogy},
\quad
\text{latent abstraction},
\quad
\text{proof dependency}.
$

Each benchmark should isolate one or two primitives before later campaigns
combine them.

## Completed axis: structural counterfactual reasoning

`counterfactual-reasoning-v1` tested a new reasoning primitive:

$$
G
\rightarrow
G'
\rightarrow
\text{reason inside }G'
\rightarrow
G.
$$

The base graph is learned from intervention/control evidence. A
counterfactual query removes an edge only inside an ephemeral world
interpretation.

Across 140 untouched official worlds:

- direct causal resolved precision: **1.000**;
- multi-hop resolved accuracy: **1.000**;
- counterfactual resolved accuracy: **1.000**;
- redundant-path preservation: **1.000**;
- bridge-cut success: **1.000**;
- missing-bridge unresolved rate: **1.000**;
- proof-path validity: **1.000**;
- exact state restoration: **1.000**.

The evidence fixtures were intentionally deterministic controlled direct-effect
probes. The result therefore establishes the structural reasoning primitive,
not robust real-world causal discovery.

See [COUNTERFACTUAL_REASONING_V1.md](COUNTERFACTUAL_REASONING_V1.md).

## Completed axis: belief revision under contradiction

`belief-revision-v1` tested cumulative revision of one causal belief inside the
only path $A\rightarrow B\rightarrow C$.

Every frozen gate passed across 60 untouched official cases.

The central latency result was:

| initial support | retraction latency | recovery latency |
|---:|---:|---:|
| 1 batch | **7** | **3** |
| 2 batches | **14** | **5** |
| 3 batches | **21** | **7** |

Cortex resisted one contradictory batch, passed through unresolved before
retraction, remained unresolved under weaker ambiguous contradiction, recovered
when causal evidence returned, propagated each epistemic state downstream, and
left every unrelated evidence cell unchanged.

See [BELIEF_REVISION_V1.md](BELIEF_REVISION_V1.md).

## Completed axis: structural analogy / isomorphism

`structural-analogy-v1` maintained a version space of joint entity/relation correspondences across independently renamed typed graphs.

Across untouched seeds 300–319:

- held-out resolved accuracy: **1.000**;
- coverage with 12 / 90 target facts: **0.9718**;
- coverage with 24 / 36 facts: **1.000 / 1.000**;
- asymmetric full unique-mapping rate: **1.000**;
- symmetric full ambiguity-preservation rate: **1.000**;
- near-isomorphic rejection: **1.000**;
- broken-analogy rejection: **1.000**.

The key qualitative result was that correspondence identity could remain ambiguous while all surviving mappings still agreed on a structural consequence.

See [STRUCTURAL_ANALOGY_V1.md](STRUCTURAL_ANALOGY_V1.md).

## Completed axis: latent concept induction

`latent-concept-induction-v1` removes one more piece of supplied structure:
the useful category itself is not given to Cortex.

The finite research prototype evaluates all **203** partitions of six opaque
entities under a predictive-description objective. Hidden concept labels are
used only after inference for scoring.

Across untouched official seeds 700–719:

- two-concept minimum partition F1: **1.000**;
- three-concept minimum partition F1: **1.000**;
- no-shared-concept control correctness: **1.000**;
- mean held-out gain over a flat per-entity predictor:
  **0.0556 / 0.0288 nats per outcome** for the two- and three-concept worlds;
- minimum grouped-case gain over one global concept: **0.2426 nats/outcome**;
- ambiguous-member unresolved rate: **1.000**;
- create/split/merge/retract lifecycle coverage: **1.000**.

A deliberately wrong geometry-decoy grouping reached at most **0.3333** F1,
while Cortex recovered the predictive partition at **1.000** F1.

The qualitative result is that Cortex can introduce a finite latent category
because pooling improves predictive-description economy, and can later split,
merge, or retract that category when current predictive structure changes.

See [LATENT_CONCEPT_INDUCTION_V1.md](LATENT_CONCEPT_INDUCTION_V1.md).

## Completed integration: concept to semantics

`concept-semantic-integration-v1` composes latent concept induction with the
existing semantic-grounding layer while preserving causal direction from
ontology to naming.

Across 80 untouched official cases:

- latent concept partition F1 remained **1.000**;
- raw adversarial neural naming was **0.3333**;
- Cortex semantic accuracy and coverage were **1.000 / 1.000**;
- replacing the complete label vocabulary changed no concept identities;
- a real latent concept with symmetric vocabulary evidence remained unnamed at
  **0.3333** maximum support;
- split/merge/retraction semantic lineage was exact.

See [CONCEPT_SEMANTIC_INTEGRATION_V1.md](CONCEPT_SEMANTIC_INTEGRATION_V1.md).

## Completed axis: explicit proof dependency

`explicit-proof-dependency-v1` replaces one-path provenance with an explicit
finite proof/dependency DAG.

Across 120 untouched official cases:

- active proof-DAG validity: **1.000**;
- provenance validity: **1.000**;
- shortest selected-proof correctness: **1.000**;
- independent alternate-proof survival: **1.000**;
- deeper-proof takeover after shortest-proof damage: **1.000**;
- shared-root invalidation correctness: **1.000**;
- contradictory-proof conflict handling: **1.000**;
- nested partial invalidation correctness: **1.000**;
- irrelevant-premise locality: **1.000**;
- exact explanation restoration: **1.000**.

A baseline that commits to only one proof failed every designated
alternate-support case.

The result establishes proof dependency management over supplied finite acyclic
rules, not unrestricted theorem proving.

See [PROOF_DEPENDENCY_V1.md](PROOF_DEPENDENCY_V1.md).

The next independent reasoning-map axis is **planning / replanning**.

## Common evaluation axes

As the Reasoning Map expands, the same difficulty axes should be swept across
multiple benchmark families:

- reasoning depth;
- missing evidence;
- noise;
- contradiction rate;
- entity count;
- rule count;
- distribution shift;
- memory pressure.

This will produce degradation curves rather than isolated pass/fail scores.

## Confidence and abstention

Future Reasoning Map campaigns should report not only correctness but the
quality of Cortex's epistemic behavior.

Important questions include:

- Does confidence fall before errors rise?
- Does unresolved activate near structural ambiguity?
- Are high-confidence conclusions actually better calibrated?
- Does Cortex preserve multiple hypotheses when evidence cannot distinguish
  them?

Raw accuracy alone is not sufficient.

## Comparison policy

When a reasoning axis stabilizes, compare the same inputs and information
budget against appropriate baselines such as:

- direct memory;
- hand-engineered symbolic closure;
- Bayesian models;
- SAT/SMT or theorem-proving systems where relevant;
- graph neural networks;
- Transformers.

The purpose is a **reasoning profile**, not an overall winner.

## Claim discipline

The Reasoning Map can support capability-specific statements tied to declared
fixtures.

It must not be collapsed into a claim of AGI, universal reasoning, or general
superiority to neural or symbolic methods.
