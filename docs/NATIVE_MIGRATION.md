# Native migration status

Cortex keeps the frozen Python `spec-v0.9.5` tree as the executable reference specification and runs the current stateful implementation natively in Rust behind differential tests.

The migration is **functionally complete at the composed runtime level** for the published v1.0.0-rc.2 contract.

## Native layers

- `cortex-articulation` — entity identity, nuisance transforms, binding, residual/noise state.
- `cortex-graph` — sparse relation/dependency state and intervention-sensitive causal evidence.
- `cortex-memory` — bounded historical memory and certified recompression.
- `cortex-core` — recurrence, adaptation, tensor plasticity, split/merge/novelty.
- `cortex-runtime` — complete articulation → graph → public summary → memory + continual transition.
- `cortex-python` — thin PyO3 research interface.

## Migration contract

A Rust layer is accepted only after deterministic differential tests reproduce the relevant frozen Python behavior. Compilation alone is not considered semantic equivalence.

The FFI boundary is intentionally coarse: one `CortexRuntime.step(...)` call performs the complete state transition inside Rust.

## Verified state

The current CI validates:

1. continual/plasticity golden replay;
2. articulation and memory differential parity;
3. sparse relation/causal replay;
4. full composed runtime replay;
5. Rust formatting and strict Clippy;
6. release-mode Rust tests;
7. Python 3.11 and 3.13 differential suites;
8. PyO3/wheel build;
9. frozen-reference integrity;
10. software/specification version consistency.

The frozen specification remains immutable under `reference/v0_9_5/`.

## Native performance

On the published controlled full-stack workload, the native path measured about **78.66 µs mean step time** versus **10,315.3 µs** for the frozen Python stack, with about **21.99 MiB** versus **87.43 MiB** peak RSS.

This is a workload-specific result. See `FULL_RUNTIME_BENCHMARK.md`.

## Current native bottleneck

The first scaling campaign exposed and removed a global sparse-graph snapshot/materialization cost from the hot path.

Subsequent stage attribution found articulation to account for roughly **83–95%** of attributed native time in the tested stress cases.

Fine-grained articulation attribution localized the dominant work to cyclic transform-distance evaluation. Matching plus subgroup-evidence transform scans consume roughly **85–96%** of attributed articulation work in the measured cases.

The next native optimization should therefore remain narrow and semantics-preserving: allocation-free transform-distance kernels and elimination of redundant evidence construction, followed by the same differential and attribution gates.

See:

- `NATIVE_SCALING_SWEEP_V1.md`
- `NATIVE_STAGE_ATTRIBUTION.md`
- `ARTICULATION_ATTRIBUTION.md`

## Relationship to the research frontier

The active predictive-field research line (currently v1.13-R) is **not yet the frozen native specification**.

Promotion should occur only after the research semantics stabilize:

```text
research mechanism
      ↓
new executable specification freeze
      ↓
differential contract
      ↓
Rust implementation
      ↓
native benchmark
```

The current `spec-v0.9.5` must not be silently rewritten to absorb experimental behavior.
