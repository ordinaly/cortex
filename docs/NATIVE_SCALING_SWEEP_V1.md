# Native scaling sweep v1

This document records the first controlled resource-scaling campaign for the fully composed `CortexRuntime`.

The purpose is **bottleneck identification**, not headline performance. The earlier full-stack benchmark established the large Python-to-native implementation gain; this campaign asks a different question: **what makes the native runtime expensive as Cortex state and observation structure grow?**

## Method

All measurements were produced by GitHub Actions run `35094091811` on an `ubuntu-24.04` hosted runner with Rust 1.98.1 and Python 3.13.

Each benchmark case runs in a fresh Python process through the public `CortexRuntime.step(...)` PyO3 interface. Frame generation is completed before timing. Each run contains 1,000 structured frames, with the first 200 excluded as warm-up and 800 measured. Every configuration is repeated three times with deterministic but distinct seeds; tables below report medians across repetitions.

The workload includes persistent entities, cyclic nuisance transforms, relation observations, intervention-sensitive outcomes, one scheduled local drift, fuzzy memory, and the continual/plasticity controller.

The sweep varies one resource axis at a time around a common baseline. It also records the *realized* runtime state: active entities, represented sparse relation/causal cells, memory prototypes, stored regimes, and tensor duty.

## Results

### Entity scale

The number of latent entities increases while eight detections remain visible per frame.

| latent entities | realized active entities | represented relation cells | mean step | p99 | peak RSS |
|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 28 | 57.8 µs | 86.0 µs | 24.26 MiB |
| 16 | 16 | 120 | 92.1 µs | 112.3 µs | 24.41 MiB |
| 32 | 31 | 465 | 184.6 µs | 217.1 µs | 25.63 MiB |
| 64 | 56 | 1,492 | 529.0 µs | 613.6 µs | 28.27 MiB |
| 96 | 78 | 2,860 | 1,006.3 µs | 1,105.0 µs | 31.34 MiB |

From 8 to 96 latent entities, median mean-step time rises by about **17.4×**. This axis simultaneously increases binding search space and accumulated graph state, so it is not sufficient by itself to identify which of those costs dominates.

### Simultaneously visible entities

With a 64-entity environment, the number of detections presented in each frame is increased.

| visible/frame | mean step | p99 | peak RSS |
|---:|---:|---:|---:|
| 2 | 134.9 µs | 213.0 µs | 19.44 MiB |
| 4 | 416.0 µs | 519.4 µs | 23.59 MiB |
| 8 | 528.8 µs | 608.5 µs | 28.11 MiB |
| 12 | 689.8 µs | 751.0 µs | 34.23 MiB |
| 16 | 885.8 µs | 954.6 µs | 41.52 MiB |

Increasing visible detections from 2 to 16 raises mean latency by about **6.6×**. This is expected to stress binding and the per-frame pairwise relation/causal update loops.

### Articulation feature dimension

| feature dimension | mean step | p99 | peak RSS |
|---:|---:|---:|---:|
| 8 | 93.9 µs | 115.4 µs | 22.15 MiB |
| 16 | 181.6 µs | 210.2 µs | 25.58 MiB |
| 32 | 264.7 µs | 300.9 µs | 30.71 MiB |
| 64 | 441.9 µs | 480.9 µs | 40.77 MiB |

An 8× increase in articulation feature dimension produces about a **4.7×** increase in mean latency. Feature-space work is therefore material, but it is not currently the steepest scaling surface.

### Co-visibility topology / sparse graph fill

This is the most diagnostic axis.

All cases contain 64 latent entities and exactly eight visible detections per frame. The only change is how entities are partitioned into co-visibility clusters. More clusters mean fewer distinct entity pairs can ever co-occur, reducing the accumulated sparse graph while leaving the *per-frame* visible pair count unchanged.

| co-visibility clusters | represented relation cells | relation storage fraction | mean step | p99 | peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,492 | 0.969 | 528.7 µs | 643.2 µs | 28.16 MiB |
| 2 | 973 | 0.600 | 361.8 µs | 453.0 µs | 27.05 MiB |
| 4 | 490 | 0.279 | 275.3 µs | 300.0 µs | 25.74 MiB |
| 8 | 238 | 0.122 | 258.8 µs | 285.8 µs | 25.11 MiB |

The dense-global case is about **2.0× slower** than the most clustered case even though both update the same number of visible pairs on every frame.

This isolates an accumulated-state cost rather than a local pair-update cost.

Code inspection identifies the likely source: `NativeCortexRuntime::articulation_vector()` currently obtains a full `SparseEvidenceGraph::snapshot()` on every step. `snapshot()` materializes every represented relation and causal cell into vectors and sorts both vectors. The runtime then scans those snapshots only to compute three summary counts for the 12-D articulation vector.

Therefore the hot path currently performs work proportional to total accumulated graph state even when the current frame touches only a small local subset.

This is an implementation bottleneck, not an architectural requirement.

### Configured regime budget

| budget | realized memory prototypes | realized stored regimes | mean step |
|---:|---:|---:|---:|
| 4 | 4 | 3 | 188.5 µs |
| 8 | 8 | 3 | 181.2 µs |
| 24 | 11 | 3 | 180.3 µs |
| 64 | 11 | 3 | 180.5 µs |

This workload stores only three continual regimes, so budgets of 8, 24 and 64 are effectively indistinguishable. The budget-4 case shows a small cost increase because fuzzy memory is actively constrained/compressed.

**Conclusion:** this sweep does **not** establish the scaling cost of a large realized regime library. A separate fixture must deliberately populate many valid recurrent regimes before that question can be answered.

### Tensor/refinement duty

Benchmark-only profiles force the conditional tensor system into low, default and continuously-awake modes.

| profile | measured tensor duty | tensor updates | curvature updates | rank checks | mean step |
|---|---:|---:|---:|---:|---:|
| low | 0.012 | 12 | 3 | 0 | 182.1 µs |
| default | 0.137 | 137 | 68 | 11 | 180.2 µs |
| high | 1.000 | 1,000 | 997 | 774 | 186.8 µs |

Moving from ~1% tensor duty to 100% tensor duty changes median mean-step time by only about **2.6%** on this d=12 continual representation.

This confirms the earlier profiling result: **the tensor/refinement math is not the present full-runtime bottleneck.** Replacing dense curvature with a more complicated low-rank method now would optimize the wrong part of the system.

## Bottleneck ranking from this campaign

The first target is the accidental full graph materialization performed while building the 12-D articulation vector. It introduces a global-state cost into a transition whose graph evidence update is otherwise local and sparse.

The next likely costs are entity binding / transform comparison and per-frame visible-pair processing. Feature dimension is a meaningful secondary scaling factor.

The conditional tensor/refinement machinery is currently a small fraction of full-frame cost at the canonical 12-D continual representation.

The regime-budget axis remains unresolved because this workload does not populate a large regime library.

## Recommended next optimization

The next change should be narrow and semantics-preserving:

1. maintain or expose lightweight graph state counts needed by the articulation summary (`relation_active`, `relation_resolved`, `causal_active`) without constructing/sorting an `EvidenceSnapshot`;
2. build the articulation summary directly from lightweight articulation state accessors rather than cloning full articulation snapshots;
3. retain full snapshots only for inspection, tests and external serialization;
4. rerun this exact campaign as an A/B comparison.

This should remove the measured global graph scan/copy from the hot path without changing any reasoning rule.

Only after that A/B run should binding, assignment, pair processing, SIMD, parallelism, or more sophisticated data structures be considered.

## Reproducibility

Campaign source:

```text
benchmarks/benchmark_native_scaling.py
```

The three-repetition median table is committed as:

```text
benchmarks/results/native_scaling_sweep_v1_medians.csv
```

The complete 75-run JSONL remains attached to GitHub Actions run `35094091811` as artifact `cortex-native-scaling-results` (artifact ID `10445611186`). Keeping the full hosted-runner output as the immutable Actions artifact avoids bloating the source tree while preserving the raw measurements.

The temporary one-shot workflow used to obtain the hosted-runner measurements should be removed after the campaign record is complete, so ordinary development pushes do not repeatedly consume benchmark resources.
