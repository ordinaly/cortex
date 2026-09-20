# Cortex v1.15-R — Cache-Refresh Attribution Result

Status: **passed engineering attribution milestone**.

Protocol: **cache-refresh-attribution-v1.15-R**.

Official GitHub Actions run: **35500122758**.

## Objective

v1.14-R shifted the dominant measured fused-stage cost from perceptual
responsibility computation to cache refresh.

v1.15-R decomposes cache refresh before choosing another optimization.

No internal winner was preregistered.

## Exactness

The profiled cache was paired against the existing sparse predictive cache.

Across all official cases:

- maximum candidate-prediction discrepancy: **0**;
- maximum local-prediction discrepancy: **0**;
- maximum outcome-count discrepancy: **0**;
- maximum cached-feature discrepancy: **0**;
- maximum cached-probability discrepancy: **0**;
- maximum predictive-distance discrepancy: **0**;
- maximum kernel discrepancy: **0**;
- structural disagreements: **0**.

The instrumentation therefore reproduced the existing cache behavior exactly
for the tested traces.

## Official attribution

| group | 72 anchors | 144 anchors | 288 anchors |
|---|---:|---:|---:|
| distance-row recomputation | **27.35%** | **33.62%** | **41.10%** |
| kernel regeneration | 21.34% | 32.68% | **35.50%** |
| writes / bookkeeping | 25.03% | 18.55% | 15.48% |
| semantic features | 9.68% | 5.45% | 2.90% |
| dirty detection | 5.70% | 3.44% | 1.88% |
| outcome probabilities | 4.23% | 2.81% | 1.55% |

At 288 anchors, the two numerical geometry stages account for:

**distance rows + kernel regeneration = 76.61% of attributed refresh time.**

The median accounting fraction at 288 anchors was **98.44%**.

## Sparse-refresh duty

Every official refresh event had at least one dirty anchor under the declared
trace.

However, the median fraction of rows updated on a dirty event decreased with
scale:

- 72 anchors: **20.54%**;
- 144 anchors: **17.19%**;
- 288 anchors: **13.40%**.

This is important: sparse maintenance is working at the row-selection level,
but the cost of recomputing each selected row against the full anchor bank
still dominates as the bank grows.

## Scaling interpretation

The attribution changes materially with scale.

At 72 anchors, writes/bookkeeping are comparable to distance computation.
By 288 anchors, numerical geometry dominates strongly.

This supports an approximately dirty-rows × all-anchors scaling interpretation
for the current geometry refresh path, followed by regeneration across every
candidate resolution.

## Next exact optimization candidate

Semantic feature rows are square roots of categorical probability vectors.
Therefore each row has unit Euclidean norm up to floating-point precision.

The current distance update constructs a three-dimensional difference tensor
for dirty rows:

```math
d_{ij}
=
\frac{\lVert f_i-f_j\rVert_2}{\sqrt{2}}.
```

For unit-norm feature rows this is algebraically equivalent to:

```math
d_{ij}
=
\sqrt{\max(0,1-f_i^\top f_j)}.
```

That suggests a v1.16-R exact semantic-geometry fast path:

1. compute dirty-row dot products against the full feature matrix;
2. derive distances from `1 - dot` without materializing a dirty × anchors ×
   outcomes difference tensor;
3. regenerate all resolution kernels from the same squared-distance quantity,
   ideally in one vectorized operation;
4. preserve the existing cached distance matrix and kernel semantics;
5. validate paired numerical and structural parity before accepting any speedup.

This directly attacks the two measured dominant groups rather than the small
feature/probability stages.

## What not to optimize next

The official measurement argues against prioritizing:

- outcome-probability refresh (**1.55%** at 288 anchors);
- semantic-feature recomputation (**2.90%**);
- dirty detection (**1.88%**).

Even eliminating all three would recover only a small fraction of the current
cache-refresh cost.

## Claim boundary

This result characterizes the current Python/NumPy research implementation on
the declared synthetic traces.

It does not prove the same bottleneck ordering for a Rust implementation, other
hardware, larger outcome dimensionality, or external workloads.

## Reproducibility

- research/cache_refresh_attribution_campaign.py
- research/cache_refresh_attribution_gate.py
- tests/test_cache_refresh_attribution.py
- docs/CACHE_REFRESH_ATTRIBUTION_V1_15.md
- benchmarks/results/cache_refresh_attribution_v1_15_summary.json
