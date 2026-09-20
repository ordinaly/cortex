# Noisy semantic revision v1 — official result

Status: **official campaign passed**.

Protocol:
[NOISY_SEMANTIC_REVISION_V1_PROTOCOL.md](NOISY_SEMANTIC_REVISION_V1_PROTOCOL.md)

Aggregate result:
[`benchmarks/results/noisy_semantic_revision_v1_summary.json`](../benchmarks/results/noisy_semantic_revision_v1_summary.json)

## Question

Can Cortex preserve useful confidence and semantic persistence when neural and
structural evidence are stochastic, incomplete, intermittently contradictory,
and changing in reliability, while still revising when the underlying semantic
regime genuinely changes?

## Frozen official campaign

Development used seeds `0..23`. After the mechanism and gates were frozen,
untouched official seeds `500..519` were evaluated across the same three
asymmetric semantic families for **60 official cases**.

The official campaign passed every frozen gate.

## Noise conditions

Each epoch included:

- 10% neural observation dropout;
- 18% wrong neural top labels among emitted observations;
- varying neural confidence and evidence reliability;
- only 35% of possible typed structural facts observed;
- 8% corruption among observed structural facts;
- changing structural reliability;
- an intermittent contradiction phase before the real regime shift;
- a simultaneous two-label semantic regime change followed by recovery.

## Main results

| Metric | Neural / baseline | Cortex |
|---|---:|---:|
| top-label accuracy | **0.8061** | **0.9639** |
| multiclass Brier score | **0.3563** | **0.0689** |
| top-support ECE | — | **0.0357** |
| trusted accepted-label accuracy | — | **0.9980** |
| trusted coverage | — | **0.7807** |
| trusted-support ECE | — | **0.00191** |
| false-certainty rate | — | **0.00195** |
| uncertainty-interception rate | — | **0.9750** |

Cortex therefore improved the finite-fixture semantic estimate substantially
over the raw noisy neural proposals while keeping trusted predictions much more
conservative than total posterior coverage.

## Revision behavior

Across all 60 official cases:

- initial accepted mapping correctness: **1.000**;
- cases with no false pre-shift semantic revision: **1.000**;
- true shift revision success: **1.000**;
- p95 shift latency: **7 epochs**;
- recovery success: **1.000**;
- p95 recovery latency: **7 epochs**;
- unaffected-concept preservation: **1.000**;
- bounded neural-history compliance: **1.000**;
- bounded structural-history compliance: **1.000**.

No official locality failure or false-revision case occurred.

## Confidence semantics

The experiment separates three surfaces:

1. neural perceptual confidence;
2. Cortex structural mapping support;
3. trust in the currently accepted semantic belief.

A transient current-epoch perceptual conflict can suspend the `trusted` flag
without immediately deleting the accepted label.

This allows:

$$
\text{accepted and trusted}
\rightarrow
\text{accepted but not trusted}
\rightarrow
\text{challenged / unresolved}
\rightarrow
\text{revised and trusted}.
$$

That distinction is central to the result. Cortex can retain semantic memory
during temporary disagreement while exposing that the memory should not
currently be treated as reliable.

## Initial acceptance under noise

Development exposed an important failure mode: accepting the first resolved
mapping can turn early noise into a persistent belief that must later be
"revised."

The accepted noisy profile therefore requires three consecutive resolved epochs
for the same candidate before initial semantic acceptance.

This removed all false-revision/locality failures in the final development and
official campaigns without modifying the already published clean
`semantic-revision-v1` defaults.

## Negative development result retained

An intermediate attempt required direct current-epoch perceptual support to
increment every withdrawal counter. That mechanism reduced real
shift/recovery adaptability without fixing the root locality failure, because
the observed failures originated during premature initial establishment.

The mechanism was rejected before protocol freeze. The successful correction
was stronger initial acceptance, not slower revision after establishment.

## Claim boundary

This result supports noisy semantic confidence and temporal revision inside the
declared finite synthetic streams.

It does not establish:

- general real-world probability calibration;
- open-vocabulary semantic understanding;
- robustness to adversarially optimized inputs;
- optimal changepoint detection or forgetting;
- correctness of arbitrary neural perception;
- trainable Cortex-to-neural representation feedback.

The next scientific integration remains to feed this grounding/revision stack
concepts that Cortex discovers through latent concept induction rather than
concept identities supplied by the fixture.
