# Native migration status

Cortex keeps the frozen Python v0.9.5 tree as the executable research specification and migrates stateful reasoning into Rust behind differential tests.

## Native layers

- `cortex-core` — continual regime reasoning, adaptive recurrence, tensor plasticity, split/merge/novelty.
- `cortex-articulation` — entity identity, cyclic nuisance transforms, soft transform evidence, subgroup synchronization, injective simultaneous binding, residual/noise state.
- `cortex-memory` — bounded Fuzzy Accordion Memory, alpha-cut activation, certified recompression, explicit unresolved/budget pressure.
- `cortex-graph` — sparse relation/dependency storage; relation and causal sufficient-stat integration remains the next migration gate.
- `cortex-python` — thin PyO3 research interface. Experiment orchestration, datasets, plotting, and reports remain Python by design.

## Migration rule

A Rust layer is not considered migrated merely because it compiles. It must reproduce the frozen Python behavioral contract on deterministic differential fixtures before it becomes part of the default native execution path.

The intended execution boundary is coarse-grained: one Python call should perform a complete state transition inside Rust rather than crossing the FFI boundary for individual numerical operations.

## Next gates

1. Land articulation and fuzzy-memory differential parity.
2. Move relation/causal sufficient statistics onto the sparse `cortex-graph` carrier.
3. Integrate articulation → graph → memory → continual plasticity under one native `step` transition.
4. Benchmark Python reference vs hybrid vs fully native execution before further optimization.
