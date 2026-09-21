# Articulation Controller Calibration v0.2

Status: **experimental learned metacognitive policy**.

## Goal

Articulation Controller v0.1 uses transparent hand-selected thresholds. v0.2
tests whether Cortex can replace those thresholds with small learned decision
surfaces while preserving interpretability and epistemic safety.

The learned controller still does **not** learn relation semantics. It learns
only when a structural state is ready for exact execution, when conflict should
cause abstention, and when the LPU is computationally preferable.

## What remains a human policy choice

Calibration cannot eliminate every normative choice.

In particular, the system needs a declared execution-risk contract: how stable
must future evidence be before exact execution is considered safe?

v0.2 makes that contract explicit and learns the decision boundary around it.

The calibration oracle uses held-out future evidence with:

- future relation reliability floor: 0.82;
- future source reliability floor: 0.80;
- future structural-support floor: 0.55;
- current 95% Wilson lower-bound floor: 0.70.

These values define the experiment's risk tolerance. They are not learned
facts about cognition.

## Future-evidence oracle

Random episodes draw latent relation reliability, source reliability, structural
support, and current evidence counts.

The controller only receives current structural features.

The calibration oracle additionally receives a large future sample from the same
latent regime.

Labels are generated as follows:

```text
future regime safe + current evidence statistically resolved
    -> EXECUTE

future regime safe + current evidence not yet resolved
    -> EXPLORE

future regime unsafe + current evidence excludes the safe region
    -> UNRESOLVED

otherwise
    -> EXPLORE
```

This prevents the learned surface from merely imitating v0.1's thresholds.

## Learned surfaces

v0.2 fits three deterministic monotone logistic surfaces.

### Execution surface

Features:

```text
evidence maturity
rule confidence
rule margin
mean structural support
source/entity support
source certainty = 1 - entropy
```

All fitted weights are constrained non-negative. More evidence or certainty
therefore cannot perversely reduce execution readiness.

The probability threshold is selected on a validation split to satisfy a high
precision target before maximizing recall.

### Conflict surface

If execution is not admitted, a second surface distinguishes persistent
conflict from continued exploration.

Features:

```text
evidence maturity
competitor confidence
1 - positive margin
1 - target confidence
source certainty
```

This captures the intended distinction:

```text
not enough information -> EXPLORE
enough incompatible information -> UNRESOLVED
```

### Backend surface

The LPU-selection surface is calibrated independently from measured execution
cost.

Features:

```text
path scale
path-compression ratio
```

Training labels come from repeated timings of the Cortex enumerating executor
versus the layered LPU executor on relation-name-agnostic lattices.

The backend probability threshold is selected to minimize measured validation
runtime regret, not merely classification error.

An explicit exact-witness request remains a hard LPU override.

## Train / validation / test discipline

Epistemic calibration episodes and backend structures are split before
threshold selection.

The final learned bundle is then tested again on native Cortex worlds with new
random seeds and opaque relation names.

The native validation is separate from the synthetic future-evidence oracle.

## Safety invariants retained

Learning the surface does not grant the controller new semantic authority.

The following remain architectural invariants:

- relation names are absent from calibration features;
- LPU cannot influence epistemic confidence;
- exact execution cannot occur without a learned execute admission;
- multi-valued targets remain valid;
- execution is read-only with respect to Cortex memory;
- an exact-witness request may choose LPU only after epistemic admission;
- resolution filtering remains upstream of exact execution.

## Why monotone linear surfaces first

A neural metacontroller would be more expressive but would make the execution
boundary harder to audit.

The first learned controller therefore uses low-capacity monotone surfaces.
If these fail on broader regimes, additional expressivity should be introduced
only alongside explicit calibration and monotonicity tests.
