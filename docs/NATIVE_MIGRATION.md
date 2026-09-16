# Native migration status

Cortex keeps the frozen Python v0.9.5 tree as the executable research specification and migrates stateful reasoning into Rust behind differential tests.

## Native layers

- `cortex-core` — continual regime reasoning, adaptive recurrence, tensor plasticity, split/merge/novelty.
- `cortex-articulation` — entity identity, cyclic nuisance transforms, soft transform evidence, subgroup synchronization, injective simultaneous binding, residual/noise state.
- `cortex-memory` — bounded Fuzzy Accordion Memory, alpha-cut activation, certified recompression, explicit unresolved/budget pressure.
- `cortex-graph` — sparse relation/dependency storage plus lazy relation and intervention-sensitive causal sufficient statistics. Pairs are allocated only after receiving evidence rather than from an entity-capacity Cartesian product.
- `cortex-runtime` — coarse-grained composition of articulation → sparse graph evidence → 12-D articulation summary → fuzzy memory + continual plasticity.
- `cortex-python` — thin PyO3 research interface. Experiment orchestration, datasets, plotting, and reports remain Python by design.

## Migration rule

A Rust layer is not considered migrated merely because it compiles. It must reproduce the frozen Python behavioral contract on deterministic differential fixtures before it becomes part of the default native execution path.

The execution boundary is deliberately coarse-grained: one Python call performs a complete state transition inside Rust rather than crossing the FFI boundary for individual numerical operations.

## Verified gates

1. Continual/plasticity core: frozen v0.9.5 golden replay.
2. Articulation and fuzzy memory: frozen v0.7/v0.8 differential parity.
3. Sparse relation/causal evidence: 320-frame frozen v0.7 replay, including posterior counts and state transitions, with no up-front O(N^2) pair allocation.
4. Full composed runtime: frozen articulation → memory → v0.9.5 path replayed frame-by-frame through one native `CortexRuntime.step(...)` call, including bindings, subgroup state, graph state changes, 12-D articulation vector, memory state, continual predictions, structural decisions, and final snapshots.

The full-runtime gate passes on both Python 3.11 and 3.13 together with canonical rustfmt, strict Clippy, Rust release tests, PyO3 compilation, and locked wheel packaging.

## First full-stack performance result

A controlled GitHub Actions campaign compared frozen Python, a hybrid native-continual configuration, and the fully native runtime using identical pre-materialized structured frames.

Median across three independent process runs:

| Mode | Mean step | p50 | Peak RSS |
|---|---:|---:|---:|
| Frozen Python | 10,315.3 µs | 9,863.6 µs | 87.43 MiB |
| Hybrid | 10,141.9 µs | 9,719.4 µs | 86.93 MiB |
| Fully native | **78.66 µs** | **72.63 µs** | **21.99 MiB** |

On that workload the fully native runtime was approximately 131× faster by median mean-step time than the frozen Python stack, while preserving the same final prediction/state fingerprint. This is a workload-specific empirical result, not a universal speedup claim. See `docs/FULL_RUNTIME_BENCHMARK.md` for methodology, raw measurements, caveats, and tail latency.

The near-flat Python → hybrid result is itself informative: the dominant full-frame overhead on this benchmark lived in the Python articulation/binding/relation-memory path, not in the already-optimized continual controller. Moving the complete stateful transition behind one native boundary removed that bottleneck.

## Next gates

1. Land the full-runtime migration and freeze its end-to-end differential fixture.
2. Run a controlled scaling campaign over entity count, visibility, graph density, feature dimension, regime budget, and tensor duty cycle.
3. Measure p50/p95/p99 latency, peak/state memory, and semantic correctness together.
4. Profile the native runtime before introducing further optimization; do not add low-rank, SIMD, parallel, or alternate graph machinery unless a measured bottleneck justifies it.
5. After the scaling envelope is declared, move to untouched external structured benchmarks for v1.0 scientific validation.
