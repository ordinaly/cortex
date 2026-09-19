# Cortex Reasoning Map v1 — Latent Concept Induction

Status: **frozen protocol / official campaign not yet run**.

Protocol ID: \`latent-concept-induction-v1\`.

## Research question

Can Cortex introduce a latent grouping that was never supplied because the
grouping compresses predictive structure?

The benchmark deliberately removes geometric clustering as the main signal.

Entities are identified only by opaque tokens and categorical predictive
evidence.

## World

Every fixture contains:

- 6 entities;
- 6 categorical outcomes;
- a matrix of observed entity/outcome counts.

The hidden generator may contain shared predictive concepts, but hidden concept
labels are used only for scoring.

Cortex receives no concept labels.

## Candidate articulation grammar

The admissible latent structures are all set partitions of the 6 entities.

There are exactly:

$$
B_6 = 203
$$

candidate partitions.

This grammar is frozen before evaluation.

No candidate partition is privileged by its hidden semantic meaning.

## Predictive model for a partition

For each candidate concept group $g$, pool the outcome counts of all entities
assigned to that group.

With Jeffreys-style smoothing

$$
\alpha = 0.5,
$$

the group outcome model is

$$
P(k\mid g)
=
\frac{n_{gk}+\alpha}
     {N_g+\alpha K}.
$$

Each entity inherits the predictive distribution of its candidate concept.

## Frozen predictive-compression objective

For partition $\pi$:

$$
\mathcal L(\pi)
=
-\sum_{e,k}
n_{ek}
\log P(k\mid \pi(e))
+
\frac{|\pi|(K-1)}{2}
\log N.
$$

The first term is predictive fit.

The second term is a BIC-style description-complexity penalty.

No free penalty coefficient is tuned after results.

Cortex retains every partition within

$$
10^{-9}
$$

of the minimum score.

This set is the latent-concept version space.

## Epistemic semantics

If exactly one partition minimizes the objective, the latent articulation is
resolved.

If several partitions tie, Cortex does not choose one arbitrarily.

For entity pair $(i,j)$, co-membership support is

$$
\mu_{ij}
=
\frac{
|\{\pi:\pi(i)=\pi(j)\}|
}{
|\mathcal V|
}.
$$

For an entity's predictive outcome distribution, Cortex resolves only when all
best partitions induce the same distribution.

Thus:

$$
\text{uncertain category membership}
\neq
\text{uncertain prediction}.
$$

## Fixtures

Official fixtures are independently relabeled by seed.

### two-concept

Three entities share one predictive law and three share another.

Training evidence is deliberately sparse and slightly uneven within each hidden
concept, so pooling should improve held-out predictive log loss relative to a
flat per-entity model.

Expected latent partition:

$$
3+3.
$$

### three-concept

Three pairs share three distinct predictive laws.

Expected partition:

$$
2+2+2.
$$

### no-shared-concept control

Each entity has a distinct outcome law.

The minimum-description solution should preserve all six singleton concepts.

This tests false concept invention.

### ambiguous-member

Five entities support two concepts.

The sixth entity has no evidence.

Assigning that entity to either established concept produces exactly the same
objective.

Required behavior:

- more than one optimal partition;
- the sixth entity remains membership-unresolved;
- Cortex does not create a fake point estimate.

## Held-out predictive test

For the two- and three-concept worlds, a larger held-out count matrix is
generated from the hidden predictive laws.

Compare:

1. Cortex latent partition;
2. flat per-entity predictor;
3. global one-concept predictor.

Report held-out negative log loss per outcome.

The primary sample-efficiency question is whether pooling under the induced
latent concept improves prediction over the flat sparse estimator.

## Structural revision diagnostic

A separate deterministic three-snapshot sequence tests reversibility.

### Base

The evidence supports two concepts:

$$
\{0,1,2\},
\qquad
\{3,4,5\}.
$$

### Split

Entity 2 acquires a distinct predictive law.

Expected partition:

$$
\{0,1\},
\{2\},
\{3,4,5\}.
$$

### Reconverged

Entity 2 again shares the first predictive law.

Expected partition returns to:

$$
\{0,1,2\},
\{3,4,5\}.
$$

v1 treats these as successive current-evidence snapshots. It tests structural
re-estimation and reversibility, not long-memory temporal filtering.

## Label anonymization

For each seed:

- entity tokens are independently permuted;
- outcome tokens/columns are independently permuted.

The learner never receives a canonical entity order or outcome meaning.

## Official seeds

Preflight uses seeds below 100.

The untouched official campaign uses:

$$
400,\ldots,419.
$$

## Metrics

Per seed:

- selected concept count;
- number of optimal partitions;
- permutation-invariant pairwise partition F1;
- held-out NLL;
- held-out NLL improvement over flat;
- held-out NLL improvement over global;
- false concept invention on singleton control;
- ambiguous-member unresolved behavior;
- split partition F1;
- reconverged partition F1.

Campaign aggregates report means and worst cases.

## Frozen gates

The official campaign passes only if:

### Concept recovery

1. two-concept pairwise F1 = **1.0** on every seed;
2. three-concept pairwise F1 = **1.0** on every seed;
3. no-shared-concept control selects 6 singleton groups on every seed.

### Predictive compression

4. mean two-concept held-out NLL improvement over flat >= **0.10 nats/outcome**;
5. mean three-concept held-out NLL improvement over flat >= **0.08 nats/outcome**;
6. latent predictor beats the one-global-concept predictor on every grouped
   seed.

### Epistemic ambiguity

7. ambiguous-member version-space size >= **2** on every seed;
8. ambiguous entity predictive state is unresolved on every seed;
9. no arbitrary unique partition is emitted for the ambiguous fixture.

### Reversibility

10. base partition F1 = **1.0**;
11. split partition F1 = **1.0**;
12. reconverged partition F1 = **1.0**.

### Leakage / grammar

13. exactly 203 candidate partitions are considered;
14. hidden concept labels are never passed into the inducer.

No gate may be weakened after official results are observed.

## Baselines

### Flat per-entity model

One independently smoothed outcome distribution per entity.

This has maximum representational freedom but weak sample pooling.

### Global one-concept model

All entities share one predictive distribution.

This compresses strongly but underfits genuinely distinct concepts.

Cortex must justify an intermediate articulation from evidence.

## Claim boundary

A passing result would support:

> Within the declared finite predictive fixtures and frozen partition grammar,
> Cortex can introduce a latent category because it improves
> predictive-description economy, preserve exact ambiguity when evidence cannot
> assign a member, and revise the latent partition when predictive structure
> splits or reconverges.

It would not establish unrestricted concept formation, semantic understanding,
or scalable unsupervised ontology learning.

## Planned files

- \`research/latent_concept_induction.py\`;
- \`research/latent_concept_induction_campaign.py\`;
- \`research/latent_concept_induction_gate.py\`;
- \`tests/test_latent_concept_induction.py\`.

The official campaign must not run until preflight is green.
