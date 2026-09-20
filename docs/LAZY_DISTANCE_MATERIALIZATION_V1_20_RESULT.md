# Cortex v1.20-R — Lazy Distance Materialization Result

Status: **passed exact optimization milestone**.

Protocol: **lazy-distance-materialization-v1.20-R**.

Official GitHub Actions run: **35505783301**.

## What changed

The research cache no longer needs to materialize ordinary predictive
distances on every reasoning step.

The validated candidate keeps squared semantic geometry on the hot path,
regenerates exactly the same predictive kernels, and marks the public distance
view stale. An explicit read of `cache.distances` reconstructs the synchronized
matrix from the current cached features.

This removes two operations from ordinary refresh:

- the distance square root;
- symmetric writes into the full distance matrix.

## Official exactness

Across all untouched official cases:

- candidate/local prediction error: **0**;
- outcome-count error: **0**;
- cached feature/probability error: **0**;
- cached kernel error: **0**;
- distance-view error after explicit materialization: **0**;
- dirty-mask disagreements: **0**;
- structural disagreements: **0**;
- hot-path distance materializations: **0**.

The public distance view therefore remains exact on demand.

## Official performance

| anchors | refresh speedup | fused-step speedup | explicit distance access |
|---:|---:|---:|---:|
| 72 | **1.073×** | **1.024×** | 77 µs |
| 144 | **1.091×** | **1.025×** | 204 µs |
| 288 | **1.132×** | **1.025×** | 780 µs |

All frozen performance gates passed.

At 288 anchors, lazy refresh is about **665.84 µs/step** versus an inferred
paired eager value of about **753.50 µs/step**, saving roughly **87.67 µs**
per ordinary refresh.

The one-time 780 µs distance materialization is therefore amortized after
roughly **9 ordinary refreshes** between explicit distance-view reads.

## Promotion

The validated lazy cache is promoted to the default **research** fused
controller. This does not modify the frozen `spec-v0.9.5` contract or the
stable Rust runtime.

The eager `ExactSemanticGeometryPredictiveCache` remains available as a
reference/oracle.

## Next engineering step

Re-run the low-overhead top-level stage profile after promotion. v1.20 changes
the cost of the formerly dominant cache stage enough that the next target
should again be selected from the new coarse ordering rather than inferred
from stale percentages.

That post-v1.20 profile should determine whether:

- cache refresh remains first, motivating kernel-write/layout work;
- candidate readout becomes the next bottleneck;
- or cache, candidate readout, hysteresis, and complexity are close enough
  that further Python micro-optimization should stop before a new spec/Rust
  promotion cycle.

## Claim boundary

This result validates lazy distance materialization for the current
Python/NumPy research path and declared synthetic traces. It does not imply
equal benefit for the stable Rust runtime or workloads that read the full
distance matrix frequently.
