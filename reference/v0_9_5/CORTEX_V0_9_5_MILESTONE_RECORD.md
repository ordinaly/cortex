# Cortex v0.9.5-alpha Milestone Record

**Name:** Compute-Optimized Refinement Runtime

**Status:** alpha; benchmarked, not frozen.

## Accepted results
1. Cached prototype-carrier representation eliminates repeated centroid-matrix reconstruction.
2. Curvature surveillance can be sampled at stride 2 with decay correction while preserving all targeted structural-action contracts in the current campaign.
3. Event-directed bookkeeping removes unnecessary split/merge maintenance on inactive paths.
4. Six real streams show zero structural and prediction mismatches against v0.9.4.
5. Weighted latency improves from 46.26 to 42.08 us/step (~1.10x speedup), recovering most of v0.9.4's refinement overhead.
6. Curvature outer-product updates fall by ~48% on the real-stream campaign.

## Non-claims
- This does not change Cortex's reasoning semantics.
- This does not prove asymptotic superiority.
- Strided curvature surveillance is not exactly identical to the dense v0.9.4 monitor on diagnostic split timing.
- The dense curvature matrix still scales as O(d^2); only its update frequency is reduced.

## Next candidate
If higher-dimensional workloads justify it, replace the dense residual-curvature matrix with a low-rank streaming sketch/subspace tracker and require the same prospective split veto before structural edits.
