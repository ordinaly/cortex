# Cortex v1.22-R — Single-Buffer Kernel Refresh Result

Status: **development rejected / official campaign not run**.

Protocol ID: **single-buffer-kernel-refresh-v1.22-R**.

## Result

The single-buffer allocation-elision candidate preserved behavior exactly but
did not improve performance on the frozen development campaign.

Across all 18 development cases:

- maximum candidate-prediction error: **0**;
- maximum local-prediction error: **0**;
- maximum outcome-count error: **0**;
- maximum feature error: **0**;
- maximum probability error: **0**;
- maximum kernel error: **0**;
- maximum materialized-distance error: **0**;
- dirty-mask disagreements: **0**;
- structural disagreements: **0**.

The lazy public-distance contract was also preserved, with zero hot-path
distance materializations.

## Performance

| anchors | refresh speedup | fused-step speedup | fused wins |
|---:|---:|---:|---:|
| 72 | **0.963×** | **0.989×** | 0 / 6 |
| 144 | **0.969×** | **0.991×** | 1 / 6 |
| 288 | **0.990×** | **0.988×** | 0 / 6 |

At 288 anchors, median refresh time moved from **665.63 µs/step** for the
validated v1.20 lazy cache to **672.58 µs/step** for the candidate. Median
complete fused-step time moved from **2497.47 µs/step** to **2534.86 µs/step**.

The frozen large-scale promotion gates required:

- refresh speedup >= **1.02×**;
- fused-step speedup >= **1.01×**;
- fused wins on at least **60%** of seeds.

All three failed.

## Interpretation

The candidate successfully removes the temporary
resolution-by-dirty-row-by-anchor kernel block and replaces it with a reusable
two-dimensional buffer. That memory-layout change is therefore valid, but it
does not pay for itself on the tested NumPy path.

The measurements are consistent with the extra per-resolution dispatch and
looping cost outweighing the allocation saved by the candidate. This is an
engineering interpretation, not a separate causal attribution claim.

The result is useful because it rules out the current allocation-elision idea
as the next profitable optimization.

## Decision

The candidate is **not promoted**.

The default fused research controller remains on
LazyDistanceSemanticGeometryPredictiveCache.

Because the development campaign failed the frozen performance gates, the
official seeds **1200–1204 were not run**. They remain untouched.

The candidate, tests, frozen protocol, and aggregate development result are
retained for reproducibility.

## Roadmap consequence

v1.21-R already showed that cache refresh remains the largest coarse stage, but
v1.22-R shows that removing this particular temporary allocation is not the
answer.

If engineering optimization continues, the next candidate should come from a
different coarse stage or a genuinely different cache algorithm rather than
another layout-only rewrite. Candidate readout remains the next-largest measured
stage at about **21%** of the 288-anchor fused step.

Scientifically, this is also a reasonable stopping point for the current cache
micro-optimization loop while the Reasoning Map advances.

See [SINGLE_BUFFER_KERNEL_REFRESH_V1_22.md](SINGLE_BUFFER_KERNEL_REFRESH_V1_22.md).
