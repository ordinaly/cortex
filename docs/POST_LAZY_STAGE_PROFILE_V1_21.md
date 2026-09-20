# Cortex v1.21-R — Post-Lazy Low-Overhead Stage Profile

Status: **development protocol / official campaign unopened**.

Protocol ID: **post-lazy-low-overhead-stage-profile-v1.21-R**.

## Motivation

v1.20-R promoted lazy distance materialization into the default research
fused-cache path. At 288 anchors it improved cache refresh **1.132×** and the
full fused step **1.025×** with zero numerical or structural discrepancies.

The next target must therefore be selected from the new top-level cost
structure rather than from the pre-v1.20 profile.

## Measurement method

v1.21-R intentionally reuses the v1.19-R low-overhead profiler unchanged.

Each stage is measured in its own fresh deterministic replay, with only one
target stage timed per replay, plus one separate replay for total step time.

This gives the strongest apples-to-apples comparison with v1.19-R while
substituting only the promoted v1.20-R lazy cache in the fused controller.

## Required cache state

The campaign aborts unless the default fused research controller uses
`LazyDistanceSemanticGeometryPredictiveCache`.

## Stages

- hysteresis/state scan;
- complexity refresh;
- perceptual responsibilities;
- candidate readout;
- local readout;
- tracker update;
- field update;
- cache refresh.

## Pre-v1.20 reference at 288 anchors

The untouched v1.19-R official profile measured:

- cache refresh: **37.22%**;
- candidate readout: **21.80%**;
- hysteresis/state scan: **17.60%**;
- complexity refresh: **13.41%**.

These values are historical comparison points only. They are not performance
gates and do not predetermine the v1.21 dominant stage.

## Scale sweep

Anchor counts: **72, 144, 288**.

Development seeds: **10..15**.

Reserved untouched official seeds: **1100..1104**.

## Validity gates

Before official evaluation:

1. the promoted lazy cache is the default fused research cache;
2. prediction-checksum discrepancy <= **1e-12**;
3. all stage and total timings are finite and non-negative;
4. total step timing is positive;
5. diagnostic refresh/scan fractions lie in **[0,1]**;
6. exactly the declared seeds and anchor scales are present.

There is deliberately **no preregistered dominant stage**.

## Decision rule

The untouched official campaign selects the next engineering action.

- If cache refresh remains clearly first, inspect post-lazy cache writes/layout
  using a new exact hypothesis rather than stale v1.17 internal shares.
- If candidate readout becomes first, move optimization effort there.
- If the leading stages are close enough that ordering is unstable across
  seeds/scales, stop Python micro-optimization and consider freezing the
  research implementation before the next specification/native promotion.

## Claim boundary

This campaign profiles the current Python/NumPy research path on deterministic
synthetic traces. It is an engineering attribution experiment, not a claim
about the stable runtime or external workloads.
