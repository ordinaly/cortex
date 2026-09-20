# Latent Concept Induction v1 — official result

Status: **official campaign passed**.

Protocol:
[LATENT_CONCEPT_INDUCTION_V1_PROTOCOL.md](LATENT_CONCEPT_INDUCTION_V1_PROTOCOL.md)

Aggregate result:
[`benchmarks/results/latent_concept_induction_v1_summary.json`](../benchmarks/results/latent_concept_induction_v1_summary.json)

## Question

Can Cortex invent a useful latent category from predictive evidence alone,
without being given the category labels, while avoiding false concepts and
preserving ambiguity?

## Frozen official campaign

The mechanism and gates were frozen after the 200-case development campaign.
Untouched official seeds 700–719 were then evaluated across five fixture types,
for **100 official cases**.

Every frozen gate passed.

## Concept recovery

- two-concept minimum partition F1: **1.000**;
- three-concept minimum partition F1: **1.000**;
- no-shared-concept control correctness: **1.000**.

The learner therefore recovered the planted predictive groupings in every
grouped official case and selected the six-singleton model in every control
case where no nontrivial shared concept was justified.

## Predictive compression

The induced latent partition was not rewarded merely for matching hidden
labels. It also had to improve held-out prediction.

- mean two-concept gain over the flat per-entity predictor:
  **0.0556 nats/outcome**;
- mean three-concept gain over flat:
  **0.0288 nats/outcome**;
- worst grouped-case gain over the one-global-concept predictor:
  **0.2426 nats/outcome**.

Thus the selected abstraction occupied the intended middle ground: more
compressed than six independent entity models, but more discriminating than a
single universal category.

## Geometry-decoy control

A fixed geometry-like decoy partition was deliberately misaligned with the
predictive grouping.

Its maximum partition F1 was **0.3333**, while Cortex recovered the predictive
partition at **1.000** F1 in every grouped case.

Cortex never received the decoy grouping. This control demonstrates that the
benchmark target is predictive organization rather than the supplied decoy
similarity structure.

## Ambiguity

The ambiguous-member fixture gives one entity no evidence while five entities
establish two concepts.

Across every official seed:

- multiple minimum-description partitions survived;
- the complete latent partition remained unresolved;
- the zero-evidence entity's predictive state remained unresolved.

Ambiguity-unresolved rate: **1.000**.

No arbitrary category assignment was emitted.

## Reversible concept lifecycle

The lifecycle fixture tested:

**coarse concept → split → merge → retract → stable empty state**.

Across every official seed:

- partition accuracy at every lifecycle stage: **1.000**;
- required create/split/merge/retract event coverage: **1.000**;
- final empty non-singleton concept state: **1.000**;
- repeated singleton snapshot emitted no spurious concept event.

Concept IDs persist only when the exact resolved member block persists.
Membership-changing split and merge operations create new concept identities
with explicit lineage.

## Grammar integrity

Every case evaluated exactly **203** candidate set partitions, the Bell number
for six entities.

Hidden concept labels were used only after inference for scoring.

## Interpretation

This is the first Cortex reasoning result in which a useful category itself is
not supplied to the learner.

The operative rule is:

> introduce a latent distinction only when its predictive fit justifies the
> representational complexity.

The result also demonstrates the reverse operation: when predictive evidence no
longer justifies a shared concept, Cortex can retract that concept rather than
preserving it merely because it existed before.

## Claim boundary

This result supports latent predictive concept induction inside the declared
finite six-entity partition grammar.

It does not establish unrestricted human-like concept learning, scalable
ontology induction, semantic understanding, or discovery from raw sensory
data.

The most important next integration is to feed these internally discovered
concept identities into the already validated semantic-grounding and semantic-
revision stack, replacing the externally supplied concept IDs used there.
