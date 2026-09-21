# Plasticity Frontier v1

Status: **active characterization campaign**.

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
