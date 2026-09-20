# Cortex v1.17-R — Optimized Cache-Refresh Re-Attribution

Status: **development protocol / official campaign unopened**.

Protocol ID: **optimized-cache-refresh-attribution-v1.17-R**.

## Motivation

v1.16-R made cache refresh materially faster but did not satisfy the frozen
bottleneck-shift gate: at 288 anchors, cache refresh still occupied **36.79%**
of the official profiled step.

The development timing had briefly suggested candidate readout had become
dominant, while the untouched official timing did not reproduce that shift.

The correct next engineering action is therefore another attribution pass
inside the **optimized cache**, not another assumed optimization.

## Research question

After the exact streamed semantic-geometry fast path, where does cache-refresh
time now go?

## Instrumented groups

The optimized cache is decomposed into:

1. semantic-feature recomputation;
2. outcome-probability refresh;
3. dirty-anchor detection;
4. streamed semantic geometry;
5. multi-resolution kernel regeneration;
6. writes and bookkeeping.

Raw substages remain available beneath these groups, including feature writes,
distance accumulation/finalization, distance writes, kernel computation, kernel
writes, and counter updates.

## Exact reference

The profiled implementation is paired against the current
`ExactSemanticGeometryPredictiveCache`.

Both controllers receive identical anchors, traces, resolutions, adaptive
resolution parameters, and initial state.

After every step compare:

- candidate predictions;
- local prediction;
- field outcome counts;
- cached semantic features;
- cached outcome probabilities;
- cached predictive distances;
- cached kernels;
- dirty masks;
- local-resolution indices.

The instrumentation is valid only if it preserves current v1.16 cache behavior.

## Scale sweep

Anchor counts:

- 72;
- 144;
- 288.

Development seeds: **0, 1, 2**.

Reserved untouched official seeds: **400, 401, 402**.

Each case uses the established equivalent/divergent/reconverged trace.

## Validity gates

1. all paired numerical discrepancies <= **1e-12**;
2. dirty-mask disagreements = **0**;
3. structural disagreements = **0**;
4. every reported timing is finite;
5. attribution accounting fraction >= **0.80** in every case;
6. all group shares lie in **[0,1]**;
7. group shares sum to at most **1.0** apart from floating-point tolerance;
8. dirty-event and dirty-row fractions lie in **[0,1]**;
9. exactly the declared seeds and anchor scales are present.

There is deliberately **no gate naming the dominant optimized substage**.

## Interpretation rule

The next optimization target is the largest official group at 288 anchors only
if it is also semantically separable enough to optimize without weakening the
validated behavior.

If no group dominates clearly, preserve that result and consider freezing the
research path rather than chasing marginal micro-optimizations.

## Timing caveat

Fine-grained `perf_counter_ns` instrumentation adds overhead. Stage shares and
scale trends are primary; absolute microsecond values are contextual.

## Claim boundary

This campaign characterizes only the current Python/NumPy optimized cache path
on the declared synthetic traces. It does not establish a universal Cortex
bottleneck ordering.
