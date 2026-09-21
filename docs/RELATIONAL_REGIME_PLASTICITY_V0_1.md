# Relational Regime Plasticity v0.1

Status: **experimental candidate**.

## Motivation

Plasticity Frontier v1 showed that the frozen cumulative-evidence stack notices
contradiction quickly but requires roughly 17.8 times accumulated historical
evidence before a replacement composition can execute.

Lowering the articulation-controller threshold would reduce that delay by also
making exact execution less safe. Deleting or globally decaying old evidence
would improve adaptation by sacrificing durable memory.

v0.1 instead separates two timescales.

## Architecture

For each opaque relational composition signature Cortex maintains:

1. **durable historical counts** — never decayed or erased;
2. **recent regime evidence** — exponentially weighted local evidence used only
   to detect whether the currently articulated regime still describes the
   present;
3. **accepted regime state** — the relation currently allowed to reach the
   articulation controller;
4. **archived regimes** — closed accepted regimes retained for inspection and
   recurrence.

The default recent decay is 0.90.

Acceptance dynamics use explicit hysteresis:

    BOOTSTRAP
       -> STABLE
       -> CHALLENGED
       -> UNRESOLVED
       -> STABLE(new regime)

A one-off observation cannot directly replace an accepted regime.

## Boundary with the articulation controller

The regime layer does not weaken or replace calibrated controller v0.2.

- STABLE: pass the active regime rule to v0.2.
- CHALLENGED: return EXPLORE.
- UNRESOLVED: return UNRESOLVED.
- Replacement is admitted only after recent evidence reaches the regime
  acceptance contract.

Thus temporal plasticity decides **which regime is current**; the articulation
controller still decides whether that regime is ready for exact execution.

## Historical preservation

When a new regime is accepted, the old one is appended to an archive with its
start/end times and observation counts. Global historical relation counts
continue growing and are never reset.

The design therefore treats temporal revision as a change of active
articulation, not destruction of prior topology.

## Frozen v0.1 candidate parameters

- recent decay: 0.90;
- challenge threshold: 0.40;
- challenge hysteresis: 0.05;
- challenge patience: 2 observations;
- replacement acceptance threshold: 0.90;
- replacement acceptance patience: 2 observations;
- minimum initial evidence: 4 observations;
- minimum replacement observations: 8.

These are experimental parameters, not ontology constants. The campaign must
show that their gains survive randomized regime shifts and stationary noise.

## Required campaign properties

The candidate is promoted only if it simultaneously demonstrates:

- bounded regime-switch lag as lifetime history grows;
- preserved historical evidence and archived regimes;
- no false switch under 10% and 20% stationary contradiction noise;
- no switch under short contradictory bursts up to eight observations;
- explicit exploration/unresolved states during sustained change;
- bounded recurrent switching back to an earlier regime;
- exact execution still gated by frozen controller v0.2.

A fast learner that achieves adaptation by becoming generally unstable fails
this campaign.

## Complexity boundary

This first candidate is intentionally low-capacity: one decayed evidence map,
one active relation, one candidate relation, and a small hysteretic state
machine per learned composition signature.

A full Bayesian change-point process or richer temporal topology may ultimately
be justified, but only if this simpler mechanism fails a measurable requirement.
Adding heavier machinery before that would increase runtime and conceptual
complexity without demonstrated need.
