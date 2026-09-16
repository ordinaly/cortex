# Native migration status

Cortex keeps the frozen Python v0.9.5 tree as the executable research specification and migrates stateful reasoning into Rust behind differential tests.

## Native layers

- `cortex-core` — continual regime reasoning, adaptive recurrence, tensor plasticity, split/merge/novelty.
- `cortex-articulation` — entity identity, cyclic nuisance transforms, soft transform evidence, subgroup synchronization, injective simultaneous binding, residual/noise state.
- `cortex-memory` — bounded Fuzzy Accordion Memory, alpha-cut activation, certified recompression, explicit unresolved/budget pressure.
- `cortex-graph` — sparse relation/dependency storage plus lazy relation and intervention-sensitive causal sufficient statistics. Pairs are allocated only after receiving evidence rather than from an entity-capacity Cartesian product.
- `cortex-python` — thin PyO3 research interface. Experiment orchestration, datasets, plotting, and reports remain Python by design.

## Migration rule

A Rust layer is not considered migrated merely because it compiles. It must reproduce the frozen Python behavioral contract on deterministic differential fixtures before it becomes part of the default native execution path.

The intended execution boundary is coarse-grained: one Python call should perform a complete state transition inside Rust rather than crossing the FFI boundary for individual numerical operations.

## Verified gates

1. Continual/plasticity core: frozen v0.9.5 golden replay.
2. Articulation and fuzzy memory: frozen v0.7/v0.8 differential parity.
3. Sparse relation/causal evidence: 320-frame frozen v0.7 replay, including posterior counts and state transitions, with no up-front O(N^2) pair allocation.

## Next gates

1. Integrate articulation → graph → memory → continual plasticity under one native full-frame transition.
2. Add end-to-end differential fixtures for the composed runtime rather than testing only layer boundaries.
3. Benchmark Python reference vs modular hybrid vs fully native execution, including memory footprint and p50/p95/p99 step latency.
4. Optimize only measured bottlenecks after full-stack parity is frozen.
