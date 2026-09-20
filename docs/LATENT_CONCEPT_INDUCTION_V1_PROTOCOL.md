# Cortex Reasoning Map v1 — Latent Concept Induction

Status: **development protocol / official campaign not opened**.

Protocol ID: `latent-concept-induction-v1`.

## Research question

Can Cortex introduce latent categories that were never supplied because those
categories improve predictive-description economy, while preserving ambiguity
and allowing the induced ontology to split, merge, or retract?

The benchmark deliberately separates **predictive concept formation** from
geometric clustering and semantic naming.

## Input

Each static fixture contains:

- 6 opaque entities;
- 6 anonymous categorical outcomes;
- an entity/outcome count matrix.

Hidden concept labels are used only for evaluation. They are never passed to
the inducer.

Entity rows and outcome columns are independently permuted by seed.

## Candidate grammar

The admissible ontology is the complete set-partition grammar over six
entities.

There are exactly

$$
B_6 = 203
$$

candidate partitions.

For a candidate partition (pi), entities in the same block share one
smoothed predictive outcome distribution.

## Predictive-description objective

With smoothing parameter

$$
alpha = 0.5,
$$

the pooled distribution for block (g) is

$$
P(kmid g)
=
rac{n_{gk}+alpha}
{N_g+alpha K}.
$$

Each candidate partition is scored by

$$
L(pi)
=
-sum_{e,k}
n_{ek}log P(kmid pi(e))
+
rac{|pi|(K-1)}{2}log N.
$$

The first term rewards predictive fit. The second is a BIC-style complexity
penalty on the number of represented blocks.

There is no tuned concept-count target.

## Epistemic semantics

Cortex retains every partition within (10^{-9}) of the minimum description
length.

If several partitions tie, it does not choose one arbitrarily.

A non-singleton concept block is exposed as **resolved** only when that exact
block appears in every minimum-description partition.

For an entity prediction, Cortex resolves only when all optimal partitions
induce the same predictive distribution for that entity.

Thus category uncertainty and predictive uncertainty remain distinct.

## Static fixtures

### Two-concept world

Three entities share one predictive law and three share another.

Training evidence uses 20 observations per entity. Held-out evaluation uses
300 observations per entity.

Expected hidden structure: (3+3).

### Three-concept world

Three pairs share three distinct predictive laws.

Training evidence uses 25 observations per entity.

Expected hidden structure: (2+2+2).

### No-shared-concept control

Each entity has a strongly distinct predictive law.

Training evidence uses 35 observations per entity.

The desired result is the six-singleton partition, with no nontrivial latent
concept exposed.

This directly tests false concept invention.

### Ambiguous-member control

Five entities establish two predictive concepts. The sixth entity has zero
evidence.

Assigning the sixth entity to either established concept produces the same
minimum description length.

Required behavior:

- version-space size at least 2;
- no unique partition claim;
- the sixth entity's predictive state remains unresolved.

## Geometry-decoy baseline

A fixed grouping pattern

`(0,1,0,1,0,1)`

is treated as a hypothetical geometry-derived clustering baseline.

It is intentionally misaligned with the planted predictive concepts.

Cortex never receives this decoy grouping. The benchmark reports its
permutation-invariant partition F1 only to verify that the target structure is
not recoverable by following the decoy grouping.

## Held-out predictive comparison

For grouped worlds, compare:

1. Cortex's induced latent partition;
2. a flat per-entity predictor;
3. a single global concept.

Report held-out negative log loss per outcome.

The test asks whether the induced intermediate abstraction improves sparse
prediction over the flat estimator while avoiding the global model's
underfitting.

## Reversible concept lifecycle

A separate deterministic sequence tests persistent latent concept IDs.

### Epoch 0 — coarse concept

All six entities share one predictive law.

Expected latent concept: one block containing all six entities.

### Epoch 1 — split

The first three and last three entities acquire distinct predictive laws.

Expected result: the coarse concept splits into two child concepts.

### Epoch 2 — merge

The two predictive laws reconverge.

Expected result: the child concepts merge into one new concept.

### Epoch 3 — retract

All six entities acquire distinct predictive laws.

Expected result: no non-singleton latent concept remains.

### Epoch 4 — persistence check

The singleton state is repeated.

Expected result: no new event is emitted and the empty latent-concept state is
stable.

Persistent concept IDs are retained only when the exact resolved member block
persists unchanged. Membership-changing split/merge operations produce new
concept IDs and explicit lineage events.

## Development and official separation

Development seeds: `0..39`.

Reserved untouched official seeds: `700..719`.

The official range must not be evaluated until development is green and the
protocol/gates are frozen.

## Development gates

The current development gates require:

### Recovery

1. minimum two-concept partition F1 = **1.0**;
2. minimum three-concept partition F1 = **1.0**;
3. singleton-control correctness = **1.0**.

### Predictive compression

4. mean two-concept held-out improvement over flat >= **0.03
   nats/outcome**;
5. mean three-concept held-out improvement over flat >= **0.015
   nats/outcome**;
6. latent predictor beats the one-global-concept predictor on every grouped
   case.

### Decoy rejection

7. geometry-decoy partition F1 never exceeds **1/3**.

### Ambiguity

8. every ambiguous-member case retains multiple optimal partitions;
9. the ambiguous entity remains prediction-unresolved.

### Lifecycle

10. coarse, split, merge, and retract partitions all achieve F1 = **1.0**;
11. `created`, `split`, `merge`, and `retracted` lifecycle events are
    all observed;
12. the final active non-singleton concept set is empty and remains empty on
    the repeated snapshot.

### Grammar integrity

13. every case evaluates exactly **203** candidate partitions.

These are development gates. They may be revised only before official
evaluation and only in response to development evidence. Once frozen, any
post-official change requires a new protocol version and fresh untouched seeds.

## Claim boundary

A passing official campaign would support the narrow claim that, inside the
declared finite predictive fixtures, Cortex can infer a useful latent
partition from predictive-description economy, preserve exact ambiguity, avoid
inventing categories in a no-shared-concept control, outperform flat/global
predictive baselines where pooling is justified, and maintain a reversible
create/split/merge/retract concept lifecycle.

It would not establish unrestricted semantic concept formation, scalable
ontology induction, natural-language meaning, or real-world unsupervised
discovery.
