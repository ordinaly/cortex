# Native stage attribution — Cortex v1.0.0-rc.2

This document records the first stage-level attribution campaign for the composed native Cortex runtime and the accompanying explicit regime-budget saturation campaign.

## Provenance

The results in this document were produced with:

- Cortex release: `1.0.0-rc.2`
- Python package version: `1.0.0rc2`
- native Rust version: `1.0.0-rc.2`
- frozen executable specification: `spec-v0.9.5`
- stage protocol: `stage-attribution-v1`
- saturation protocol: `regime-saturation-v1`
- benchmarked Git commit: `079f7816af7cf87495dd710c7f53d9af3167277d`
- Python runtime: 3.13.15
- repetitions: 3 per case

The protocol identifiers are independent of the Cortex software version. Re-running either protocol on another Cortex release should retain the protocol identifier unless the workload or measurement semantics change.

## Why this campaign exists

The first native scaling sweep identified full graph snapshot materialization as an accidental per-frame cost. After that cost was removed, the next question was no longer whether native Cortex was fast in aggregate, but **which stage now owns the remaining frame time**.

The composed transition was therefore instrumented at its native boundaries:

```text
articulation / entity binding / invariance
        ↓
NodeId binding conversion
        ↓
sparse relation + causal graph observation
        ↓
12-D public articulation summary
        ↓
fuzzy historical memory
        ↓
continual regime reasoning / plasticity
```

The normal production `CortexRuntime.step()` remains uninstrumented. The campaign uses a separate opt-in `profile_step()` path that executes the same state transition and records native `Instant` timings around each stage. A Rust parity test verifies that the profiled and unprofiled paths produce the same relevant state transition.

## Stage-attribution-v1 results

The table below reports the median of three fresh-process repetitions for each deterministic case. Stage percentages use the sum of attributed native stages as the denominator.

| case | internal mean | articulation | graph | public summary | continual | dominant stage |
|---|---:|---:|---:|---:|---:|---|
| baseline, 32 entities / 8 visible / 16-D | 126.92 µs | 113.53 µs (89.6%) | 6.62 µs (5.2%) | 5.06 µs (4.0%) | 0.71 µs (0.6%) | articulation |
| 96 nominal entities / 8 visible | 281.00 µs | 258.72 µs (92.2%) | 7.75 µs (2.8%) | 12.84 µs (4.6%) | 0.70 µs (0.3%) | articulation |
| 64 nominal / 16 visible | 442.11 µs | 376.63 µs (85.8%) | 50.71 µs (11.6%) | 10.06 µs (2.3%) | 0.79 µs (0.2%) | articulation |
| 64-D features | 384.51 µs | 365.33 µs (95.1%) | 6.82 µs (1.8%) | 10.27 µs (2.7%) | 0.87 µs (0.2%) | articulation |
| sparse topology, 8 co-visibility clusters | 229.36 µs | 201.71 µs (86.4%) | 6.50 µs (2.8%) | 23.55 µs (10.1%) | 0.78 µs (0.3%) | articulation |
| high tensor duty | 131.63 µs | 111.46 µs (82.9%) | 6.66 µs (5.0%) | 5.07 µs (3.8%) | 10.42 µs (7.8%) | articulation |

Binding conversion and fuzzy-memory stages remain below 1% in every case and are omitted from the compact table; their values are retained in the committed CSV.

### Main finding

**Articulation is now the dominant native cost in every tested regime, accounting for approximately 83–95% of attributed frame time.**

This remains true under three deliberately different stressors:

- increasing represented entity population to 96 raises articulation to about 92% of attributed time;
- increasing feature dimensionality from 16 to 64 raises articulation to about 95%;
- increasing visible entities to 16 makes graph pair processing visible at about 12%, but articulation still owns about 86%.

The high-tensor-duty case is also informative. It drives the continual controller from well below 1% to about 7.8% of attributed time, yet articulation remains the dominant stage. This supports retaining conditional tensor execution while directing general optimization effort elsewhere.

### Secondary finding: public-summary construction

The public-summary stage is normally small, but rises to about 10% in the sparse-topology case. The current summary path still obtains an articulation snapshot, which clones more articulation state than the 12-D summary strictly requires. This resembles the graph-snapshot issue removed in the previous optimization pass, although its absolute cost is much smaller. It is a valid secondary optimization target after the dominant articulation path has been decomposed.

## What stage-attribution-v1 does not yet tell us

The campaign establishes *which top-level layer* dominates, but it does not yet separate the internal articulation costs. The current articulation transition contains several candidates:

- entity × detection × nuisance-transform scoring;
- construction of the simultaneous binding cost matrix;
- linear-sum assignment;
- transform-evidence / subgroup-evidence updates;
- prototype and reliability updates;
- residual/noise-state updates.

The observed scaling with entity population and feature dimension is consistent with work in those paths, but this campaign does not justify assigning a specific percentage to any one of them.

The next profiling protocol should therefore be `articulation-attribution-v1`, instrumenting these internal stages before introducing SIMD, parallelism, caching, low-rank approximations, or a different assignment algorithm.

## Regime-saturation-v1

The original native scaling campaign varied the configured regime budget, but typical workloads realized only two or three regimes. That did not actually stress the bounded-state contract. `regime-saturation-v1` fixes that by presenting deliberately separated 12-D regimes until the budget is full and then continuing with four additional novel regimes.

Tensor, split, and merge machinery is suppressed in this synthetic fixture so the experiment isolates budget behavior. Six observations are supplied per attempted regime.

| budget | attempted regimes | stored regimes | first pressure step | budget-pressure frames | unresolved frames | median mean step |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6 | 2 | 13 | 24 | 24 | 1.14 µs |
| 4 | 8 | 4 | 25 | 24 | 24 | 1.17 µs |
| 8 | 12 | 8 | 49 | 24 | 24 | 1.27 µs |
| 16 | 20 | 16 | 97 | 24 | 24 | 1.57 µs |

The onset steps are exactly `budget × 6 + 1`: the first frame belonging to the first regime beyond capacity.

### Bounded-state result

For every tested budget:

- stored regime count reaches the configured budget exactly;
- it never grows beyond that budget;
- the first post-capacity novel regime immediately exposes `budget_pressure` and `unresolved` rather than silently allocating another prototype;
- all four additional regimes remain represented as explicit pressure/unresolved observations rather than hidden structural growth.

This is an engineering validation of the implemented bounded-state contract under this synthetic fixture. It is not a proof that every possible stream satisfies all desired theoretical boundedness properties.

The comparison count grows with the configured budget (median totals 66, 156, 408, and 1200 across the four cases), while step latency remains in the low-microsecond range for this isolated continual-core fixture. The full composed runtime remains dominated by articulation instead.

## Reproducible result files

Committed median summaries:

- `benchmarks/results/stage_attribution_v1_medians.csv`
- `benchmarks/results/regime_saturation_v1_medians.csv`

The benchmark scripts themselves are:

- `benchmarks/benchmark_stage_attribution.py`
- `benchmarks/benchmark_regime_saturation.py`

Both use `benchmarks/provenance.py` to write the software version, native version, Python package version, frozen specification version, protocol identifier, exact Git SHA, Python runtime, seeds, and repetition metadata into every raw result row.

## Engineering decision

The next performance work should **not** begin with broad parallelization or general-purpose SIMD. The measured sequence is now:

1. decompose articulation with `articulation-attribution-v1`;
2. remove any accidental allocation/copying work found there while preserving the frozen differential contract;
3. rerun `stage-attribution-v1` as the A/B gate;
4. only then consider algorithmic or hardware-level optimization of a demonstrated hot substage.

This maintains the Cortex optimization rule established by the native migration: optimize measured bottlenecks without weakening semantics or adding complexity speculatively.
