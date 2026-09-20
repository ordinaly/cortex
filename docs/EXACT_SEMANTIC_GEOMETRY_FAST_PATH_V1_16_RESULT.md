# Cortex v1.16-R — Exact Semantic-Geometry Fast Path Result

Status: **exact optimization validated; frozen milestone did not pass all gates**.

Protocol: **exact-semantic-geometry-fast-path-v1.16-R**.

Official GitHub Actions run: **35501085877**.

## What changed

The accepted implementation avoids the dirty-rows × anchors × outcomes
difference tensor used by the legacy sparse cache. It accumulates squared
feature differences directly into a 2-D dirty-row matrix and reuses that same
squared geometry for every candidate-resolution kernel.

The default fused research controller now uses
`ExactSemanticGeometryPredictiveCache`. The previous
`SupportSparsePredictiveCache` remains available as the paired reference.

## Development correction retained

The first development implementation used the unit-norm identity

```math
d_{ij}^2 = 1-f_i^\top f_j.
```

It was faster but produced a maximum cached-distance discrepancy of
**2.33e-9** near zero because the square root amplifies cancellation in
`1 - dot`.

That implementation was rejected rather than relaxing the distance-parity
gate.

The final implementation uses streamed component-wise squared differences,
which preserves the legacy Euclidean arithmetic much more closely while still
avoiding the 3-D temporary.

## Official exactness

Across all official cases:

- candidate-prediction error: **2.22e-16**;
- local-prediction error: **1.11e-16**;
- outcome-count error: **0**;
- cached feature/probability error: **0**;
- cached distance error: **5.55e-17**;
- cached kernel error: **3.33e-16**;
- dirty-mask disagreements: **0**;
- structural disagreements: **0**.

## Official performance

| anchors | refresh speedup | fused-step speedup | cache-refresh share |
|---:|---:|---:|---:|
| 72 | **1.18×** | **1.06×** | 33.57% |
| 144 | **1.34×** | **1.12×** | 36.40% |
| 288 | **1.52×** | **1.17×** | **36.79%** |

The direct speedup gates therefore passed.

## Frozen gate that failed

The official protocol required cache-refresh share at 288 anchors to fall to
at most **35%** of the instrumented fused step.

The observed median was **36.79%**.

Therefore the frozen milestone does **not** pass every gate.

This threshold is intentionally not weakened after evaluation.

## Interpretation

The correct conclusion is narrower than a full v1.16 pass:

- the implementation is an exact and useful optimization;
- cache refresh becomes materially faster;
- the complete fused step also becomes materially faster;
- however, cache refresh remains the largest measured official stage.

The development run briefly suggested candidate readout had become dominant,
but the untouched official run did not reproduce that bottleneck shift.

That discrepancy is itself useful evidence: stage ordering is close enough
that the next target should be chosen from a fresh decomposition rather than
from development timing alone.

## Best next engineering experiment

Re-profile the **optimized** cache internals before changing them again.

The next campaign should attribute the remaining refresh cost among:

- semantic-feature recomputation;
- outcome-probability refresh;
- dirty detection;
- streamed squared-distance accumulation;
- multi-resolution kernel regeneration;
- cache writes/bookkeeping.

This is preferable to immediately optimizing candidate readout because the
official v1.16 result still measures cache refresh as the largest stage.

## Claim boundary

This result supports an exact Python/NumPy cache-refresh optimization on the
declared synthetic workloads.

It does not support the preregistered claim that cache refresh falls below
35% of the full profiled step, and it does not imply the same effect in Rust
or on external workloads.
