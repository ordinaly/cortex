# Plasticity Frontier v1

Status: **completed baseline characterization**.

Plasticity is a core Cortex requirement, but it is not one scalar property.
This campaign decomposes it into measurable behaviors before introducing a new
adaptation mechanism.

## Questions

1. How quickly does a stable learned relation stop being executable after the
   environment changes?
2. How much new evidence is required before a replacement relation becomes
   executable?
3. Does adaptation latency grow with accumulated historical evidence?
4. Does a short contradictory burst destabilize a mature regularity?
5. Does revision remain local to the affected relational structure?
6. Can the system recover after a transient contradiction?

## Current mechanism under test

The present relational composition learner is cumulative:

    confidence = examples_for_relation / examples_for_signature

There is no temporal discount, regime segmentation, or source reliability in
that ratio. Old and new observations have equal permanent weight.

The native relation graph likewise accumulates Beta evidence without temporal
decay. Fuzzy accordion memory is different: it already decays prototype utility
and can perform certified recompression.

This means Cortex currently contains multiple stability/plasticity timescales
rather than one global plasticity coefficient.

## Regime-shift protocol

For each randomized world, Cortex first observes an opaque composition

    R0 ; R1 -> OLD

for N solved examples, with N in:

    4, 8, 16, 32, 64

A held-out path is never supplied with its answer.

The environment then changes and independently supplies:

    R0 ; R1 -> NEW

one example at a time.

At every step we measure:

- OLD calibrated state;
- NEW calibrated state;
- fixed-v0.1 state;
- OLD and NEW composition confidence;
- the state and confidence of a disjoint control composition.

Primary transition markers are:

- first point OLD leaves EXECUTE;
- first point OLD becomes UNRESOLVED;
- first point NEW reaches EXECUTE;
- replacement evidence divided by prior evidence.

If replacement evidence grows approximately linearly with prior evidence, the
system has measurable **historical inertia**: old experience creates a
plasticity debt even when the environment has genuinely changed.

## Noise-burst protocol

A mature 32-example composition receives a contradictory burst of 1, 2, 4, or
8 examples and then returns to the original regime.

We record the deepest controller state reached and how much supporting evidence
is required to return to EXECUTE.

This distinguishes useful stability against transient noise from rigidity under
a persistent regime change.

## Locality control

A disjoint eight-example composition is present in every regime-shift world.
Its confidence must remain 1.0 and its calibrated controller state must remain
EXECUTE throughout the unrelated shift.

## Claim boundary

This benchmark diagnoses the current stability/plasticity frontier. It does not
yet alter Cortex semantics.

A temporal or regime-aware mechanism should only be introduced after this
baseline demonstrates a concrete failure mode and gives us a metric it must
improve without increasing catastrophic forgetting.


## First characterization result

The first seven-seed sweep exposed a strong and highly regular historical
inertia in the frozen v0.2 policy.

| prior OLD examples | leave OLD EXECUTE | OLD UNRESOLVED | NEW reaches EXECUTE | NEW / prior |
|---:|---:|---:|---:|---:|
| 4 | 1 | 3 | 72 | 18.000 |
| 8 | 1 | 5 | 143 | 17.875 |
| 16 | 1 | 9 | 285 | 17.813 |
| 32 | 2 | 18 | 570 | 17.813 |
| 64 | 4 | 36 | 1139 | 17.797 |

The replacement cost therefore scales almost exactly with accumulated history:

    replacement evidence ~= 17.8 * prior evidence

This is not merely failure to detect novelty. The calibrated controller notices
change quickly: OLD leaves EXECUTE after only about 5-6% contradictory evidence.
The long delay occurs between detecting that the old articulation is unsafe and
admitting the replacement articulation.

The result is a large **re-articulation dead zone**.

### Transient contradiction

With a mature 32-example relation:

| contradictory burst | deepest state | supporting observations to EXECUTE |
|---:|---|---:|
| 1 | EXECUTE | 0 |
| 2 | EXPLORE | 4 |
| 4 | EXPLORE | 40 |
| 8 | EXPLORE | 111 |

The controller therefore has useful resistance to a single contradictory event,
but recovery cost rises sharply once a short burst moves the confidence margin
away from the very conservative execution boundary.

### Locality

The disjoint control composition remained at confidence 1.0 and state EXECUTE
throughout every regime-shift trajectory. The observed rigidity is local rather
than catastrophic forgetting of unrelated structure.

## Diagnosis

Two mechanisms combine to produce the measured frontier.

1. Composition evidence is cumulative and gives every historical example
   permanent equal weight.
2. The calibrated v0.2 execution surface requires a very large positive relation
   margin before exact execution is admitted.

This combination has a desirable property: Cortex stops making an exact claim
early when evidence starts to disagree.

But it has an undesirable property: once a genuine regime change has occurred,
the new regime must numerically overwhelm the entire old history before Cortex
will articulate it exactly.

The next experiment should therefore **not delete old evidence and should not
simply lower the execution threshold**. Either change would sacrifice the
stability that the controller is successfully providing.

## Candidate direction — dual-timescale regime evidence

The next prototype should maintain two distinguishable quantities for each
learned relational pattern:

- **historical evidence**: slow, durable, provenance-preserving memory;
- **current-regime evidence**: fast, recency-sensitive evidence used to detect
  and characterize a possible world change.

A regime transition should be explicit and hysteretic:

    stable articulation
        -> change suspected
        -> conflict / unresolved
        -> new stable articulation

Old evidence remains available as historical structure rather than being
overwritten.

This is closer to online change-point reasoning than naive forgetting. A simple
exponential-decay rule will still be benchmarked as a baseline, but it should
not be promoted merely because it adapts quickly: it must also resist stationary
noise and preserve recoverable history.

## Promotion criteria for a plasticity mechanism

A candidate should improve all of the following simultaneously:

- replacement lag should be bounded rather than proportional to total lifetime
  evidence;
- one-off and low-rate contradictory noise should not cause false regime
  switches;
- persistent change should move through an explicit uncertainty state before
  re-execution;
- switching back should be possible without reconstructing erased history;
- unrelated relations should remain unchanged;
- old regimes should remain inspectable as historical evidence;
- controller false-execution risk must not exceed the frozen v0.2 contract;
- memory/runtime growth must remain bounded.

The benchmark should reject a mechanism that gains plasticity only by making
Cortex generally less stable.


## Follow-up — Relational Regime Plasticity v0.1

The dual-timescale follow-up has now passed its Pareto calibration and CI, and its selected knee is frozen as the research default.
It preserves the exact v0.2 contradiction-detection lags while bounding the
replacement switch to **9 observations** across all tested prior masses
4 / 8 / 16 / 32 / 64, rather than 72 / 143 / 285 / 570 / 1139.

The selected research policy retains exact historical evidence, uses a fast
recency-sensitive regime channel, and adds explicit hysteretic suspicion and
confirmation. It produced zero declared false switches under transient bursts
and stationary-noise controls, reversed OLD -> NEW -> OLD in 9 observations per
switch, and kept fixed-vocabulary regime state bounded with stream length.

See [RELATIONAL_REGIME_PLASTICITY_V0_1.md](RELATIONAL_REGIME_PLASTICITY_V0_1.md).
