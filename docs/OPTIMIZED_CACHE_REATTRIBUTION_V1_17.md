# Cortex v1.17-R — Optimized Cache Re-Attribution

Status: **development protocol / official campaign unopened**.

Protocol ID: **optimized-cache-reattribution-v1.17-R**.

## Motivation

v1.16-R made the cache-refresh path materially faster but did not satisfy the
frozen bottleneck-shift gate: cache refresh still occupied **36.79%** of the
official profiled fused step at 288 anchors.

The development profile briefly suggested candidate readout had become
dominant, while the untouched official profile still placed cache refresh
first. The stage ordering is therefore too close to choose another
optimization from stale attribution.

v1.17-R re-attributes the **optimized** cache before selecting another target.

## Question

Where does cache-refresh time go after the v1.16 streamed-geometry
optimization?

## Optimized-cache attribution groups

1. semantic-feature recomputation;
2. outcome-probability refresh;
3. dirty-anchor detection;
4. streamed semantic geometry;
5. multi-resolution kernel regeneration;
6. writes and bookkeeping.

The streamed-geometry group retains separate raw timers for:

- squared-distance accumulation;
- final distance square root.

This prevents the summary grouping from hiding a different internal bottleneck.

## Exact reference

The profiled cache subclasses the current
`ExactSemanticGeometryPredictiveCache` and reproduces its refresh path with
timing boundaries only.

A paired controller using the uninstrumented optimized cache receives the same
anchors, observations, outcomes, resolutions, and local-resolution state.

After every step compare:

- candidate predictions;
- local prediction;
- field outcome counts;
- cached semantic features;
- cached outcome probabilities;
- cached predictive distances;
- cached kernels;
- dirty-anchor masks;
- local-resolution indices.

## Scale sweep

Anchor counts: **72, 144, 288**.

Development seeds: **0, 1, 2**.

Reserved untouched official seeds: **400, 401, 402**.

Each case uses the existing equivalent/divergent/reconverged deterministic
trace.

## Metrics

For each scale report median:

- total optimized-cache refresh microseconds per step;
- share and microseconds per step for each attribution group;
- raw squared-distance accumulation share;
- raw distance-square-root share;
- dirty-event fraction;
- mean dirty-row fraction;
- explicit accounting fraction.

## Frozen-intent validity gates

Before official evaluation the development implementation must satisfy:

1. all paired prediction/cache discrepancies <= **1e-12**;
2. dirty-mask disagreements = **0**;
3. structural disagreements = **0**;
4. every reported timing is finite;
5. explicit accounting fraction >= **0.80** for every case;
6. every group share lies in **[0,1]**;
7. group shares sum to at most **1.0** apart from floating-point tolerance;
8. dirty-event and dirty-row fractions lie in **[0,1]**;
9. exactly the declared development seeds and anchor scales are present.

There is deliberately **no preregistered dominant group**.

The official campaign determines the next engineering target.

## Interpretation rule

A follow-up optimization is justified only if a group remains a substantial
share at 288 anchors and admits an exact or explicitly bounded transformation.

If candidate readout outside the cache has become larger than every cache
substage, v1.17-R should end the cache micro-optimization sequence rather than
forcing another cache change.

## Timing caveat

Fine-grained timing adds overhead. Stage shares, scale trends, and relative
ordering are the primary quantities; absolute microseconds are secondary.

## Claim boundary

This campaign attributes the current Python/NumPy optimized cache path on the
declared synthetic traces. It does not establish a universal bottleneck for
the Rust runtime or external workloads.
