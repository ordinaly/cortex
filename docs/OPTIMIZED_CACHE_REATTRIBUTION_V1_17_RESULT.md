# Cortex v1.17-R — Optimized Cache Re-Attribution Result

Status: **passed engineering attribution milestone**.

Protocol: **optimized-cache-reattribution-v1.17-R**.

Official GitHub Actions run: **35503268072**.

## Objective

v1.16-R made semantic geometry exact and faster, but cache refresh still
occupied 36.79% of the official profiled fused step. Development and official
stage ordering also disagreed on whether candidate readout had become dominant.

v1.17-R therefore re-attributes the **optimized** cache before selecting
another micro-optimization.

## Exactness

The timed cache reproduced the current `ExactSemanticGeometryPredictiveCache`
exactly on every official trace:

- candidate/local prediction discrepancy: **0**;
- outcome-count discrepancy: **0**;
- feature/probability discrepancy: **0**;
- distance discrepancy: **0**;
- kernel discrepancy: **0**;
- dirty-mask disagreements: **0**;
- structural disagreements: **0**.

## Official attribution

| group | 72 anchors | 144 anchors | 288 anchors |
|---|---:|---:|---:|
| kernel regeneration | 18.23% | **37.25%** | **48.63%** |
| writes / bookkeeping | **32.67%** | 26.34% | **24.94%** |
| streamed semantic geometry | 16.82% | 14.95% | 13.69% |
| semantic features | 10.84% | 7.20% | 4.31% |
| dirty detection | 6.70% | 4.55% | 2.82% |
| outcome probabilities | 4.85% | 3.74% | 2.32% |

At 288 anchors, kernel regeneration is now by far the largest remaining
cache-refresh computation.

## Geometry substage

At 288 anchors, the streamed-geometry group decomposes into:

- squared-distance accumulation: **11.40%**;
- distance square root: **2.21%**.

This confirms that the v1.16 geometry rewrite materially reduced the geometry
share. Another geometry-focused optimization is not justified by the current
profile.

## Sparse-refresh duty

Every official event had at least one dirty anchor. The median dirty-row
fraction fell with scale and was **13.28%** at 288 anchors.

The optimized sparse selection therefore remains effective; the dominant cost
is what must be regenerated for each selected row.

## Next exact optimization candidate

The current kernel for resolution r is

```math
K_r
=
\exp\left(-\frac{d^2}{2r^2}\right).
```

For the canonical resolution ladder 0.03, 0.06, 0.12, 0.25, the first three
resolutions have exact squared-scale ratios:

```math
\alpha_{0.03}=16\alpha_{0.12},
\qquad
\alpha_{0.06}=4\alpha_{0.12},
```

where `alpha_r = 1 / (2 r^2)`.

Therefore the 0.12 kernel can be exponentiated once and the 0.06 / 0.03
kernels can, in principle, be derived by exact integer powers:

```math
K_{0.06}=K_{0.12}^{4},
\qquad
K_{0.03}=K_{0.12}^{16}.
```

The 0.25 kernel still requires an independent exponential.

A v1.18-R experiment should test this **kernel-ladder fast path** against the
current direct four-exponential implementation. It must preserve kernel and
prediction parity under the existing strict gates; underflow and accumulated
roundoff must be measured rather than assumed safe.

## Secondary target

Writes/bookkeeping remain **24.94%** of optimized refresh, with kernel writes
alone about **16.39%**. If the kernel-ladder experiment succeeds, cache-memory
layout or write fusion is the next likely target.

## Independent replication caveat

A parallel historical v1.17 attribution branch (PR #54) replayed the same
v1.16 optimized-cache semantics with exact behavioral parity but produced a
different fine-grained timing split on its untouched official run.

At 288 anchors that independent run measured:

- writes/bookkeeping: **44.29%**;
- streamed semantic geometry: **20.74%**;
- kernel regeneration: **15.48%**.

This contrasts with the consolidated v1.17 run, where kernel regeneration
appeared dominant. The disagreement is retained as evidence that fine-grained
within-refresh timing is sensitive to instrumentation and runner conditions.

That instability is consistent with the later v1.18 result, where optimizing
kernel regeneration failed to reproduce the large end-to-end gain suggested by
the original internal attribution. PR #54 is therefore preserved as historical
evidence but should not be merged over the consolidated research line.

## Claim boundary

This result characterizes the current Python/NumPy optimized cache on the
declared synthetic traces. It does not imply the same ordering in Rust or on
external workloads.

## Reproducibility

- `research/optimized_cache_reattribution_campaign.py`
- `research/optimized_cache_reattribution_gate.py`
- `tests/test_optimized_cache_reattribution.py`
- `docs/OPTIMIZED_CACHE_REATTRIBUTION_V1_17.md`
- `benchmarks/results/optimized_cache_reattribution_v1_17_summary.json`
