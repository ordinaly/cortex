# Cortex Reasoning Map v1 — Structural Analogy / Isomorphism

Status: **frozen protocol / official campaign passed**.

Protocol ID: `structural-analogy-v1`.

This is the third dedicated experiment in **Cortex Reasoning Map v1** after
structural counterfactual reasoning and belief revision.

## Research question

Can Cortex recognize and exploit the same typed relational organization when
both entity labels and relation labels are independently replaced?

The intended abstraction is:

$$
G_s
\xrightarrow{\text{structural correspondence}}
G_t
$$

without relying on shared surface symbols.

## Scientific boundary

v1 tests finite typed-graph analogy under partial explicit structural evidence.

It does not establish:

- unrestricted semantic analogy;
- natural-language analogy;
- approximate large-scale graph matching;
- latent relation discovery;
- superiority over exact graph-isomorphism algorithms.

An exact graph-isomorphism solver is an algorithmic oracle/reference here, not
a competitor Cortex is expected to outperform.

## Worlds

Every source world contains:

- 6 entities;
- 3 directed relation types;
- a finite typed edge set.

For every seed, source and target use disjoint opaque namespaces.

Entity labels are independently permuted:

$$
S_i \notin \{T_j\},
$$

and relation labels are independently permuted:

$$
R_i^{(s)} \notin \{R_j^{(t)}\}.
$$

The hidden source-to-target entity bijection and relation bijection are never
exposed to the inducer.

## Families

Four canonical relational families are used.

### asymmetric-chain

An asymmetric typed dependency chain with cross-links.

Purpose:

- unique full structural correspondence;
- relation-label recovery.

### asymmetric-branch

A second asymmetric topology with different local degree signatures.

Purpose:

- prevent overfitting to one motif.

### symmetric-twins

Two entities occupy exactly interchangeable structural roles.

Purpose:

- node automorphism;
- mapping ambiguity must remain explicit.

### symmetric-diamond

A typed diamond contains an exact branch-swap automorphism.

Purpose:

- distinguish uncertain correspondence from certain relational consequences.

For the two asymmetric families, complete target evidence must admit exactly one
joint node/relation correspondence.

For the symmetric families, complete target evidence must admit more than one.

## Partial target evidence

The source graph is fully available.

The target graph is only partially observed.

Evidence is ternary structural data:

$$
(u,r,v,\text{present})
$$

or

$$
(u,r,v,\text{absent}).
$$

The target contains 90 possible directed non-self triples:

$$
6\times 5\times 3=90.
$$

Evidence is revealed in a deterministic nested order with positive and negative
facts interleaved.

Official budgets are:

$$
12,\quad 24,\quad 36
$$

observed target triples.

Because the evidence sets are nested, the surviving correspondence version
space must never increase as budget grows.

## Cortex correspondence state

Cortex maintains every joint bijection

$$
(\phi_V,\phi_R)
$$

that is consistent with all observed target facts.

A candidate survives only if applying the source structure through the
candidate node and relation mappings reproduces every observed target fact.

This produces a version space:

$$
\mathcal V_t.
$$

No arbitrary tie-break is allowed.

## Predictive analogy

For an unobserved target triple $q$, every surviving correspondence predicts
whether $q$ is present.

Cortex returns:

- **present** when every candidate predicts present;
- **absent** when every candidate predicts absent;
- **unresolved** when surviving candidates disagree;
- **incompatible** when the correspondence version space is empty.

Thus:

$$
|\mathcal V_t|>1
$$

does not necessarily imply predictive uncertainty.

Several correspondences may remain possible while making the same structural
prediction.

## Mapping confidence

For each source entity $s$, Cortex records the empirical candidate distribution

$$
P_t(v\mid s)
=
\frac{
|\{\phi\in\mathcal V_t:\phi_V(s)=v\}|
}{
|\mathcal V_t|
}.
$$

The correspondence is resolved only when this distribution is a point mass.

The same rule applies to relation labels.

No single scalar is interpreted as a calibrated probability of metaphysical
identity; it is version-space support.

## Held-out evaluation

At each target-evidence budget, all unobserved target triples form the held-out
query set.

Metrics include:

- held-out accuracy;
- held-out coverage;
- resolved accuracy;
- positive-edge recall among resolved queries;
- negative-fact recall among resolved queries;
- surviving candidate count;
- resolved entity correspondences;
- resolved relation correspondences.

The hidden true mapping is always one surviving candidate in compatible worlds.

Therefore a unanimous prediction should be exact; disagreement should remain
unresolved.

## Scratch baseline

A target-only direct-memory baseline knows only the revealed target facts.

Because held-out queries exclude observed facts, its held-out coverage is zero.

This baseline measures the sample-efficiency value of transferred source
structure, not algorithmic graph-isomorphism difficulty.

## Exact structural oracle

A full-evidence exact-isomorphism diagnostic receives all 90 target facts.

For asymmetric families it must recover exactly one candidate.

For symmetric families it must retain more than one candidate.

This oracle characterizes identifiability.

## Near-isomorphic negative control

For every seed, one asymmetric target is perturbed by a single typed structural
change chosen so the complete target is not isomorphic to the source.

Complete diagnostic evidence is then supplied.

Cortex must reject the analogy:

$$
\mathcal V=\varnothing.
$$

This is a safety control, not a sample-efficiency comparison.

## Broken-analogy update

A compatible partial target is first processed normally.

Then one explicit contradictory target fact is revealed that invalidates every
remaining source correspondence.

Cortex must move from a non-empty version space to **incompatible** rather than
preserving a previously useful analogy.

This tests analogy revision after disconfirming evidence.

## Ambiguity control

On the two symmetric families, complete evidence still admits structural
automorphisms.

Required behavior:

- mapping uniqueness must remain false;
- at least one entity correspondence must remain unresolved;
- held-out structural consequences may still resolve when all automorphisms
  agree.

This directly tests:

$$
\text{uncertain identity}
\neq
\text{uncertain structure}.
$$

## Provenance

Every resolved held-out prediction records:

- number of surviving correspondences;
- unanimous support count;
- structural source triple(s) implied under the surviving mappings.

v1 does not yet build a general proof DAG, but every transferred conclusion is
auditable against the active version space.

## Official seeds

Development and preflight use seeds below 100.

The untouched official campaign uses:

$$
300,\ldots,319.
$$

With 4 families and 3 evidence budgets, the compatible campaign contains:

$$
20\times4\times3=240
$$

cases.

Negative and full-identifiability diagnostics are recorded separately.

## Metrics

Campaign metrics:

- source/target entity-label overlap;
- source/target relation-label overlap;
- held-out resolved accuracy;
- held-out coverage at 12 / 24 / 36 facts;
- positive and negative resolved accuracy;
- candidate-version-space monotonicity;
- asymmetric full unique-mapping rate;
- symmetric full ambiguity-preservation rate;
- asymmetric resolved-mapping precision;
- near-isomorphic rejection rate;
- broken-analogy rejection rate;
- scratch held-out coverage;
- mean candidate count by budget.

## Frozen gates

The official campaign passes only if all of the following hold.

### Leakage

1. source/target entity-label overlap = **0**;
2. source/target relation-label overlap = **0**.

### Conservative transfer

3. held-out resolved accuracy = **1.0** at every budget;
4. positive resolved accuracy = **1.0**;
5. negative resolved accuracy = **1.0**.

### Sample efficiency

6. mean held-out coverage at 12 facts >= **0.20**;
7. mean held-out coverage at 24 facts >= **0.45**;
8. mean held-out coverage at 36 facts >= **0.65**;
9. direct-memory held-out coverage = **0.0**.

### Version-space behavior

10. candidate count is non-increasing with nested evidence on every compatible
    seed/family;
11. asymmetric full unique-mapping rate = **1.0**;
12. asymmetric resolved-mapping precision = **1.0**;
13. symmetric full ambiguity-preservation rate = **1.0**.

### Negative transfer

14. near-isomorphic complete-evidence rejection rate = **1.0**;
15. broken-analogy contradiction rejection rate = **1.0**.

No threshold may be weakened after official results are observed.

## Negative-result policy

If preflight exposes a protocol/implementation mismatch, advance the protocol
version before any official campaign.

If an official campaign fails, preserve the result and use fresh untouched
seeds for any revised official claim.

## Planned files

- `research/structural_analogy.py`
- `research/structural_analogy_campaign.py`
- `research/structural_analogy_gate.py`
- `tests/test_structural_analogy.py`

The official campaign must not run until preflight is green.

## Intended claim boundary

A passing result would support:

> Within declared finite typed-graph fixtures, Cortex can maintain a
> version-space of structural correspondences across independently renamed
> worlds, exploit unanimous structure for held-out predictions, preserve
> mapping ambiguity under automorphisms, and reject an analogy when explicit
> contradictory structure invalidates all correspondences.

It would not establish unrestricted human-like analogy or semantic transfer.


## Official outcome

The untouched official campaign on seeds 300–319 passed every frozen gate.

Key results:

- held-out resolved accuracy: **1.000**;
- mean held-out coverage at 12 facts: **0.9718**;
- coverage at 24 / 36 facts: **1.000 / 1.000**;
- asymmetric full unique-mapping rate: **1.000**;
- symmetric full ambiguity-preservation rate: **1.000**;
- near-isomorphic rejection: **1.000**;
- broken-analogy rejection: **1.000**;
- source/target entity and relation label overlap: **0**.

See [STRUCTURAL_ANALOGY_V1.md](STRUCTURAL_ANALOGY_V1.md) and
[`benchmarks/results/structural_analogy_v1_summary.json`](../benchmarks/results/structural_analogy_v1_summary.json).

The frozen thresholds above were not weakened after evaluation.
