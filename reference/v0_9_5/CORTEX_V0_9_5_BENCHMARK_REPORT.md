# Cortex v0.9.5-alpha Benchmark Report

## Objective
Measure whether v0.9.4 rank-triggered refinement can be made cheaper without losing its behavioral contracts or changing its previously validated real-stream trajectories.

## Controlled behavioral campaign
Frozen fixtures and seeds were reused from v0.9.4.

- Parametric update: 8/8 remained local, no split.
- Persistent heterogeneous concept: 8/8 promoted exactly one split.
- Genuine novel regime: 8/8 created a new regime without split.
- Split then collapse: 5/5 split and later merged back to one concept.

Mean Brier scores:

| Fixture | v0.9.4 | v0.9.5-alpha |
|---|---:|---:|
| Update | 0.248696 | 0.248696 |
| Split | 0.179303 | 0.176966 |
| Novel | 0.162824 | 0.162824 |
| Split -> merge | 0.213103 | 0.216664 |

The split and merge event times changed because the curvature monitor is sampled. The *type of structural action* was preserved in every acceptance case. The merge fixture shows a small predictive cost, so v0.9.5 remains alpha rather than frozen.

## Real-stream non-interference
Six chronological streams were replayed, totaling 3,636 observations.

- structural mismatches vs v0.9.4: 0
- prediction mismatches vs v0.9.4: 0
- promoted real-stream splits: 0 in both versions

Thus the compute optimization did not alter any previously validated real-stream output in this campaign.

## Runtime
Median per-stream timings were aggregated with stream length as weight.

| Version | Weighted latency (us/step) |
|---|---:|
| v0.9.3 tensor activation | 41.51 |
| v0.9.4 rank refinement | 46.26 |
| v0.9.5-alpha optimized refinement | 42.08 |

v0.9.5-alpha is approximately 1.10x faster than v0.9.4, recovering most of the refinement overhead. It is within about 1.4% of v0.9.3 while retaining split/merge capability.

The dense curvature update count across the six real streams falls by about 48%.

## Interpretation
The main gain came from removing repeated structural-carrier reconstruction, not from a mathematically more sophisticated eigensolver. At the current d=12, dense curvature algebra is too small to dominate runtime. This is a useful negative result: optimizing the theoretically expensive term first would have missed the actual Python/runtime bottleneck.

For larger d, the O(d^2) curvature state remains the next asymptotic cost center. A future version should evaluate low-rank streaming covariance/eigenspace sketches rather than keeping a dense Q_i.
