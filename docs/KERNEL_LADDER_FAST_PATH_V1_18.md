# Cortex v1.18-R — Exact Kernel-Ladder Fast Path

Status: **development protocol / official campaign unopened**.

Protocol ID: **kernel-ladder-fast-path-v1.18-R**.

## Motivation

v1.17-R re-attributed the optimized cache and found kernel regeneration to be
the dominant remaining cache-refresh group at 288 anchors:

- kernel regeneration: **48.63%**;
- writes/bookkeeping: **24.94%**;
- streamed semantic geometry: **13.69%**.

The canonical semantic-resolution ladder is 0.03, 0.06, 0.12, 0.25.

## Exact exponent structure

For

```math
K_r=\exp\left(-\frac{d^2}{2r^2}\right),
```

the first three resolutions have exact inverse-square ratios relative to 0.12:

```math
\frac{1}{0.03^2}:\frac{1}{0.06^2}:\frac{1}{0.12^2}
=
16:4:1.
```

Therefore:

```math
K_{0.06}=K_{0.12}^{4},
\qquad
K_{0.03}=K_{0.12}^{16}.
```

The 0.25 kernel has no corresponding integer-power relation and remains
independently exponentiated.

## Candidate implementation

The candidate allocates the final kernel tensor once.

It computes:

1. one exponential for the 0.12 kernel;
2. two repeated squarings for 0.06;
3. two more repeated squarings for 0.03;
4. one independent exponential for 0.25.

Non-canonical resolution sets fall back to the existing direct implementation.

## Reference

The current `ExactSemanticGeometryPredictiveCache` remains the direct
four-exponential oracle.

The candidate `KernelLadderSemanticGeometryPredictiveCache` is evaluated only
inside paired research controllers until the official campaign passes.

## Exactness checks

After every step compare:

- candidate and local predictions;
- field outcome counts;
- cached semantic features and probabilities;
- cached distances;
- all cached kernels;
- dirty-anchor masks;
- local-resolution indices;
- exact zero/underflow pattern of the kernel tensor.

## Performance metrics

For every case measure:

- focused kernel-regeneration time;
- full cache-refresh time;
- complete fused-step time;
- corresponding speedups over the direct kernel oracle.

Execution order alternates between reference and candidate each step to reduce
timing-order bias.

## Scale sweep

Anchor counts: **72, 144, 288**.

Development seeds: **0, 1, 2**.

Reserved untouched official seeds: **500, 501, 502**.

## Development validity gates

Before official evaluation:

1. candidate/local prediction error <= **1e-12**;
2. outcome/feature/probability/distance error <= **1e-12**;
3. kernel error <= **1e-12**;
4. kernel underflow-pattern mismatches = **0**;
5. dirty-mask disagreements = **0**;
6. structural disagreements = **0**;
7. all reported timings and speedups are finite;
8. exactly the declared development seeds and scales are present.

Development performance is used only to set a frozen official performance gate.

## Interpretation rule

If repeated powers alter underflow behavior or exceed the numerical parity
contract, reject the ladder rather than loosening the gate.

If exactness passes but performance is neutral or worse, retain that negative
result and move to the writes/bookkeeping target identified by v1.17-R.

## Claim boundary

A passing result would establish only an exact-or-machine-parity optimization
for the canonical Python/NumPy resolution ladder. It would not imply equal
benefit for arbitrary resolutions, Rust, or external workloads.
