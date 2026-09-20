# Cortex v1.20-R — Lazy Distance Materialization

Status: **development protocol / official campaign unopened**.

Protocol ID: **lazy-distance-materialization-v1.20-R**.

## Motivation

v1.19-R confirmed cache refresh as the largest coarse Python/NumPy stage at
288 anchors (**37.22%** of the fused step).

The hot predictive path consumes cached kernels directly. Candidate readout,
local readout, and complexity evaluation do not require the full cached
distance matrix on every step.

v1.20-R tests whether that diagnostic/public distance view can be synchronized
lazily without changing reasoning behavior.

## Candidate

The eager cache currently computes dirty-row squared geometry, takes a square
root to obtain ordinary distances, writes the symmetric distance matrix, then
regenerates kernels from squared geometry.

The lazy candidate:

1. computes the same dirty-row squared geometry;
2. regenerates exactly the same kernels;
3. skips the hot-path square root;
4. skips symmetric distance-cache writes;
5. marks the public distance view stale;
6. reconstructs the exact full distance matrix only when `distances` is
   explicitly accessed.

The existing eager `ExactSemanticGeometryPredictiveCache` remains the oracle.

## Public-view invariant

Lazy state is an implementation detail only.

An explicit read of `cache.distances` must return a fully synchronized matrix
matching the eager cache within the frozen numerical tolerance. Re-reading a
fresh view must not trigger another rebuild.

## Hot-path parity

During the timed replay compare after every step:

- candidate predictions;
- local prediction;
- outcome counts;
- cached semantic features;
- cached outcome probabilities;
- cached kernels;
- dirty masks;
- local-resolution indices.

The timed replay must perform **zero distance materializations**.

After timing completes, access the lazy distance view once and compare it with
the eager matrix.

## Performance metrics

For each paired case measure:

- eager versus lazy cache-refresh time;
- eager versus lazy complete fused-step time;
- explicit lazy distance-view materialization latency;
- refresh and fused-step speedup.

Diagnostic materialization latency is reported separately and excluded from
hot-path speedups.

## Scale sweep

Anchor counts: **72, 144, 288**.

Development seeds: **0..7**.

Reserved untouched official seeds: **900..904**.

## Development validity gates

Before official evaluation:

1. hot-path prediction/state/kernel discrepancies <= **1e-12**;
2. explicit distance-view discrepancy <= **1e-12**;
3. dirty-mask disagreements = **0**;
4. structural disagreements = **0**;
5. hot-path distance materializations = **0**;
6. the distance view is stale before explicit access;
7. one access performs exactly one materialization and leaves the view fresh;
8. all timing metrics are finite;
9. exactly the declared development seeds/scales are present.

Development performance determines any later frozen official speed gates.

## Interpretation rule

If the hot path does not improve measurably, retain the result and do not
promote lazy state merely for elegance.

If hot-path performance improves but explicit diagnostic access is expensive,
report both costs. Laziness is justified only for workloads where distance-view
reads are genuinely colder than reasoning steps.

## Claim boundary

A passing result would establish an implementation optimization for the current
Python/NumPy research cache. It would not change the stable runtime semantics
or imply the same benefit in Rust.
