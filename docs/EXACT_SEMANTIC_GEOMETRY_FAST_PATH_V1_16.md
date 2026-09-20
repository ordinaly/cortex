# Cortex v1.16-R — Exact Semantic-Geometry Fast Path

Status: **frozen protocol / development passed / official campaign untouched**.

Protocol ID: **exact-semantic-geometry-fast-path-v1.16-R**.

## Motivation

v1.15-R attributed the cache-refresh bottleneck at 288 anchors as:

- distance-row recomputation: **41.10%**;
- kernel regeneration: **35.50%**;
- writes/bookkeeping: **15.48%**;
- semantic features: **2.90%**;
- dirty detection: **1.88%**;
- outcome probabilities: **1.55%**.

Distance geometry plus kernel regeneration therefore account for about
**76.6%** of attributed refresh time.

v1.16-R attacks those two measured terms without changing sparse dirty-row
selection or predictive semantics.

## Exact geometry identity

The semantic feature row for one anchor is the square root of a smoothed
categorical probability vector:

```math
f_i(k)=\sqrt{p_i(k)}.
```

Therefore each feature row has unit Euclidean norm up to floating-point
roundoff:

```math
\lVert f_i\rVert_2^2
=
\sum_k p_i(k)
=
1.
```

The existing predictive distance is

```math
d_{ij}
=
\frac{\lVert f_i-f_j\rVert_2}{\sqrt{2}}.
```

For unit feature rows this implies

```math
d_{ij}^2
=
1-f_i^\top f_j.
```

The optimized cache therefore computes dirty-row similarities with a matrix
product, derives the squared distances once, and reuses them for every
resolution kernel.

Self-distances are explicitly pinned to zero to avoid square-root amplification
of floating-point error on the diagonal.

## Kernel reuse

For semantic resolution r, the current kernel is

```math
K_r(i,j)
=
\exp\left(-\frac{d_{ij}^2}{2r^2}\right).
```

The optimized path regenerates all resolution rows directly from the shared
squared-distance matrix rather than repeatedly squaring distance rows.

## Reference implementation

The legacy `SupportSparsePredictiveCache` remains available as the paired
oracle.

The default fused research controller uses the new
`ExactSemanticGeometryPredictiveCache`.

Both controllers receive identical anchors, traces, resolutions, local
adaptive-resolution parameters, and initial state.

## Paired exactness checks

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

## Performance measurements

For each paired case measure:

1. cache-refresh time;
2. complete fused prepare/update/refresh step time;
3. post-optimization fused-stage attribution.

Execution order alternates each step to reduce systematic timing-order bias.

## Scale sweep

Anchor counts:

- 72;
- 144;
- 288.

Development seeds: **0, 1, 2**.

Reserved untouched official seeds: **300, 301, 302**.

## Frozen official gates

Before any official run:

1. candidate-prediction error <= **1e-12**;
2. local-prediction error <= **1e-12**;
3. outcome-count error <= **1e-12**;
4. feature/probability error <= **1e-12**;
5. predictive-distance error <= **1e-10**;
6. kernel error <= **1e-12**;
7. dirty-mask disagreements = **0**;
8. structural disagreements = **0**;
9. fused-stage timing accounting >= **0.90**;
10. all reported metrics are finite and the declared seeds/scales are exact.

The final frozen gates additionally require, at **288 anchors**:

11. median cache-refresh speedup >= **1.30×**;
12. median complete fused-step speedup >= **1.08×**;
13. median cache-refresh stage share <= **0.35** after optimization.

These thresholds were frozen after development and before the official seeds
were opened.

## Interpretation rule

A speedup is accepted only if the exactness contract remains green.

If the dot-product formulation is slower on the current low-dimensional
semantic features, retain that negative result rather than weakening the
protocol. A mathematically elegant identity is not automatically a useful
implementation optimization.

## Claim boundary

A passing result would support an exact optimization claim for the current
Python/NumPy predictive-geometry research path.

It would not imply the same speedup for Rust, different feature dimensionality,
other BLAS implementations, or external workloads.

## Development history before freeze

The first implementation used the unit-norm dot-product identity directly.
It was fast, but near-zero cached distances suffered cancellation after the
square root. Development measured:

- cache-refresh speedup at 288 anchors: **1.65×**;
- full fused-step speedup: **1.20×**;
- maximum distance-cache error: **2.33e-9**;
- maximum kernel error: **2.16e-13**;
- structural disagreements: **0**.

The distance error violated the stricter parity gate, so that implementation
was rejected rather than weakening the threshold.

The accepted development implementation instead accumulates squared feature
differences one feature dimension at a time into a 2-D dirty-row matrix. This
avoids the 3-D difference tensor while preserving the legacy Euclidean
arithmetic much more closely. The squared-distance matrix is still reused for
all candidate-resolution kernels.

Final development result at 288 anchors:

- cache-refresh speedup: **1.5688×**;
- complete fused-step speedup: **1.1338×**;
- cache-refresh stage share: **0.2603**;
- candidate-readout stage share: **0.3656**;
- maximum distance error: **5.55e-17**;
- maximum kernel error: **3.33e-16**;
- candidate-prediction error: **2.22e-16**;
- structural disagreements: **0**.

No implementation, threshold, or official seed may now be changed without
creating a new protocol version.
