# Cortex v1.22-R — Single-Buffer Kernel Refresh

Status: **frozen protocol / candidate not promoted**.

Protocol ID: **single-buffer-kernel-refresh-v1.22-R**.

## Motivation

The v1.21-R low-overhead profile measured cache refresh as the largest coarse
Python/NumPy stage at **35.55%** of the 288-anchor fused step.

The promoted v1.20 lazy cache still regenerates dirty kernel rows by creating a
temporary array with shape approximately:

$$
R \times D \times A,
$$

where \(R\) is the number of candidate resolutions, \(D\) the dirty-row count,
and \(A\) the anchor count.

The v1.22-R candidate tests whether that temporary can be removed without
changing numerical or structural behavior.

## Candidate

SingleBufferKernelRefreshPredictiveCache inherits the validated v1.20 lazy
distance semantics.

For each dirty refresh it:

1. computes the same exact squared semantic geometry;
2. obtains one reusable two-dimensional scratch view;
3. evaluates one resolution kernel at a time into that buffer using NumPy
   output-buffer operations;
4. copies the resulting rows into the persistent kernel bank;
5. preserves symmetric writes and lazy public-distance materialization.

The scratch bank starts at 25% of the anchor count and grows geometrically only
if a later refresh needs more rows. The v1.17 official campaign measured a mean
dirty-row fraction of about 13.3%, so the initial capacity is deliberately
above the observed ordinary refresh regime while remaining much smaller than a
full anchor-by-anchor bank.

The validated LazyDistanceSemanticGeometryPredictiveCache remains the oracle
and default implementation during evaluation.

## Scale and seeds

Anchor counts:

- 72;
- 144;
- 288.

Development seeds:

- 30–35.

Untouched official seeds:

- 1200–1204.

The official seeds must not run until the development campaign has passed and
this protocol, implementation, and gates are frozen.

## Exactness gates

Across every case:

- candidate prediction error <= **1e-12**;
- local prediction error <= **1e-12**;
- outcome-count error <= **1e-12**;
- feature error <= **1e-12**;
- probability error <= **1e-12**;
- kernel error <= **1e-12**;
- explicitly materialized distance-view error <= **1e-12**;
- dirty-mask disagreements = **0**;
- structural disagreements = **0**;
- no hot-path distance materializations;
- both implementations preserve the lazy stale/fresh distance-view contract;
- the candidate scratch buffer is strictly two-dimensional and is exercised.

## Frozen performance gates

The optimization is only worth promotion if it improves the complete fused
step, not merely the internal kernel timer.

At 72 and 144 anchors:

- median fused speedup must be at least **0.98×**.

At 288 anchors:

- median cache-refresh speedup must be at least **1.02×**;
- median complete fused-step speedup must be at least **1.01×**;
- the candidate must beat the reference fused step on at least **60%** of
  seeds.

These thresholds are frozen before development and official evaluation.

## Decision rule

If every untouched official gate passes:

- promote the single-buffer cache to the default research fused controller;
- record v1.22-R as a passed exact optimization;
- run one post-promotion low-overhead profile before selecting another target.

If any official performance gate fails:

- retain the candidate for reproducibility;
- do not change the default cache;
- record the result as a negative or mixed experiment;
- stop assuming allocation elision is an economically important bottleneck.

No threshold is weakened after seeing official results.

## Claim boundary

This experiment evaluates one Python/NumPy cache implementation on the frozen
synthetic predictive-geometry traces. It makes no claim about the stable Rust
runtime or external workloads.
