# Noisy semantic revision v1 — confidence and revision under stochastic evidence

Status: **frozen protocol / official campaign passed**.

Protocol ID: `noisy-semantic-revision-v1`.

## Research question

Can Cortex preserve calibrated uncertainty and useful semantic persistence when
perception and structural evidence are noisy, incomplete, intermittently
contradictory, and changing in reliability?

The test extends the clean temporal revision result into a stochastic stream.
It asks whether Cortex becomes uncertain before it becomes confidently wrong,
while still adapting to a real semantic regime change.

## Confidence surfaces

The revision layer now exposes three distinct quantities.

1. **Perceptual confidence** — the neural candidate-label distribution.
2. **Structural posterior support** — Cortex's finite mapping marginal for each
   concept-label correspondence.
3. **Accepted-belief confidence** — support currently assigned to the label
   Cortex continues to accept after temporal hysteresis.

For an accepted label `l*`, the revision read exposes its current support,
the strongest competing label/support, and

$$
P_{rev}
=
\max(0, s_{challenger}-s_{incumbent}),
$$

called **revision pressure**.

A belief is marked `trusted` only when it remains in the stable phase and its
current support is at least the normal semantic-grounding acceptance threshold.

This means an accepted label can temporarily remain in memory while no longer
being presented as trusted.

## Noise model

Each epoch independently introduces several forms of uncertainty.

### Neural evidence

- 10% probability that a concept receives no neural observation;
- 18% probability that an emitted neural top label is wrong;
- top-label confidence varies between 0.58 and 0.90;
- evidence reliability varies over time;
- wrong neural observations are, on average, assigned lower reliability but
  can still be individually high-confidence.

### Structural evidence

For every possible typed concept relation:

- only 35% of facts are observed at a given epoch;
- 8% of observed structural facts are flipped;
- structural reliability varies by observation;
- corrupted structural facts receive a reliability penalty.

This produces both missing evidence and contradictory evidence.

## Chronological regimes

Each world contains five opaque concepts, five disjoint opaque labels, and one
of the three asymmetric semantic relation families used in the preceding
grounding/revision campaigns.

The sequence is:

- 6 establishment epochs under the base mapping;
- 10 noisy steady-state epochs;
- 6 intermittent-contradiction epochs;
- 14 epochs under a true simultaneous two-label regime shift;
- 14 noisy recovery epochs after the original mapping returns.

During intermittent contradiction, neural evidence periodically proposes the
future shifted mapping while structural evidence still supports the base
mapping at reduced reliability. The ground truth does not change in this phase.

## Core metrics

### Prediction quality

The neural and Cortex mapping distributions are compared using:

- top-label accuracy;
- multiclass Brier score.

The purpose is not to claim universal probabilistic calibration. Cortex's
mapping posterior is finite normalized support over the declared hypothesis
space. The Brier measurement asks empirically whether that support behaves
usefully on this campaign.

### Calibration

For Cortex top-label support and trusted accepted-label support, report
10-bin expected calibration error.

### Selective confidence

Report:

- trusted coverage;
- trusted accuracy;
- false-certainty rate;
- uncertainty-interception rate.

The uncertainty-interception rate is the fraction of epochs where Cortex still
carries an accepted label that is wrong under the current regime but correctly
marks that label as **not trusted**.

The desired behavior during a regime change is therefore:

$$
\text{old label trusted}
\rightarrow
\text{old label not trusted}
\rightarrow
\text{unresolved}
\rightarrow
\text{new label trusted}.
$$

### Revision behavior

Report:

- false revisions before a real regime shift;
- shifted-mapping revision success;
- revision latency;
- original-mapping recovery success;
- recovery latency;
- revisions of unaffected concepts.

### Boundedness

The same finite 12-epoch evidence horizon from semantic revision v1 remains in
force. Neural and structural history sizes are checked directly.

The clean semantic-revision controller accepts an initially resolved label
immediately. Under stochastic evidence that proved too eager: an early noisy
mapping can be accepted and later corrected, which looks like a semantic
revision even though the real regime never changed.

The noisy v1 profile therefore requires **three consecutive resolved epochs for
the same candidate before initial acceptance**.

After establishment, revision remains responsive. The noisy profile keeps the
two-conflict persistence of the clean controller, but a withdrawal counter
advances only when the long-horizon grounding posterior resolves to a concrete
competing label. Mere ambiguity can challenge trust without withdrawing the
incumbent.

Separately, a sufficiently strong current-epoch neural disagreement can suspend
the `trusted` flag immediately while the accepted label remains in memory.
This separates:

$$
\text{stop trusting}
$$

from

$$
\text{change the accepted label}.
$$

These are robustness parameters for the noisy experiment, not retroactive
changes to the already published clean semantic-revision-v1 protocol.

## Development and official separation

Development seeds are `0..23`.

Official seeds are `500..519`.

The protocol and gates were frozen before those seeds were evaluated. Any
post-official mechanism or threshold change requires a new protocol version
and fresh untouched seeds.

## Frozen official gates

The development campaign passed the following gates. They are now frozen
unchanged for the untouched official campaign:

- at least 95% correct establishment;
- Cortex top-label accuracy at least 3 percentage points above raw neural
  top-label accuracy;
- Cortex Brier score no worse than the neural baseline;
- mean Cortex top-support ECE <= 0.20;
- trusted accuracy >= 0.98;
- trusted coverage >= 0.60;
- false-certainty rate <= 0.02;
- uncertainty-interception rate >= 0.95;
- at least 95% of cases with no false semantic revision before the real shift;
- shift and recovery success >= 0.95;
- 95th-percentile shift and recovery latency <= 10 epochs;
- unaffected-concept preservation >= 0.98;
- bounded-history compliance = 1.0.

The passing 72-case development campaign measured:

- neural top-label accuracy: **0.7975**;
- Cortex top-label accuracy: **0.9631**;
- neural Brier score: **0.3686**;
- Cortex Brier score: **0.0699**;
- mean top-support ECE: **0.0364**;
- trusted accuracy: **0.9976** at **0.7755** coverage;
- false-certainty rate: **0.00236**;
- uncertainty-interception rate: **0.9715**;
- false revision cases: **0**;
- unaffected-concept preservation: **1.000**;
- shift success: **1.000**, p95 latency **7 epochs**;
- recovery success: **1.000**, p95 latency **8 epochs**.

No gate or mechanism may now be changed before official evaluation. Any change
after opening the official seed range requires a new protocol version and fresh
untouched seeds.

## Claim boundary

A passing final official campaign would support noisy semantic confidence and
revision only inside the declared finite synthetic streams.

It would not establish:

- calibrated confidence on arbitrary real-world concepts;
- robustness to adversarially optimized inputs;
- optimal forgetting or changepoint detection;
- open-vocabulary semantics;
- correctness of a neural encoder;
- trainable Cortex-to-neural feedback.


## Official outcome

The untouched 60-case official campaign on seeds 500–519 passed every frozen
gate.

Official aggregate metrics:

- neural top-label accuracy: **0.8061**;
- Cortex top-label accuracy: **0.9639**;
- neural Brier score: **0.3563**;
- Cortex Brier score: **0.0689**;
- mean Cortex top-support ECE: **0.0357**;
- trusted accepted-label accuracy: **0.9980** at **0.7807** coverage;
- trusted-support ECE: **0.00191**;
- false-certainty rate: **0.00195**;
- uncertainty-interception rate: **0.9750**;
- false pre-shift revision cases: **0**;
- shift success: **1.000**, p95 latency **7 epochs**;
- recovery success: **1.000**, p95 latency **7 epochs**;
- unaffected-concept preservation: **1.000**;
- bounded-history compliance: **1.000**.

See [NOISY_SEMANTIC_REVISION_V1.md](NOISY_SEMANTIC_REVISION_V1.md) and
[the committed aggregate summary](../benchmarks/results/noisy_semantic_revision_v1_summary.json).

The frozen gates were not weakened after official evaluation.
