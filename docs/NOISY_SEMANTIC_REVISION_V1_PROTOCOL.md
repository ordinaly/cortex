# Noisy semantic revision v1 — confidence and revision under stochastic evidence

Status: **development protocol / official campaign not yet opened**.

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

The clean semantic-revision controller uses two consecutive resolved conflicts
before withdrawing an incumbent. The noisy v1 profile intentionally raises
this to **three consecutive resolved conflicts**. This is a robustness
parameter for the noisy experiment, not a retroactive change to the already
published clean semantic-revision-v1 protocol.

## Development and official separation

Development seeds are `0..23`.

Reserved official seeds are `500..519`.

No official seed may be evaluated until the development campaign is reviewed
and the final gates are frozen. Development is explicitly allowed to expose
weaknesses and motivate changes to the mechanism or thresholds.

Any post-official change requires a new protocol version and fresh untouched
seeds.

## Current development gates

The initial development gate is intentionally conservative and tests:

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

These are development gates, not yet an official preregistration. They may be
tightened or the model may be revised using development evidence. They will not
be weakened after official evaluation begins.

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
