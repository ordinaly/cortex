# Full native runtime benchmark

This benchmark is the first end-to-end timing comparison after composing the independently differential-tested native layers behind one coarse-grained `CortexRuntime.step(...)` call.

It is a **performance characterization of one controlled workload**, not a universal speedup claim.

## Compared execution modes

- **Python** — frozen v0.8 articulation + fuzzy memory, followed by the frozen v0.9.5 continual controller.
- **Hybrid** — frozen Python v0.8 articulation + fuzzy memory, followed by the native Rust continual controller.
- **Native** — one PyO3 call into the composed Rust runtime: articulation → sparse relation/causal evidence → 12-D articulation summary → fuzzy memory + continual plasticity.

All three modes consume the same pre-materialized deterministic frame sequence and finish with the same fingerprint:

```text
prediction = 0.20916490804840276
stored regimes = 2
```

The benchmark therefore compares execution cost only after the semantic migration gates have passed.

## Workload and environment

- GitHub Actions hosted `ubuntu-24.04` runner
- Rust 1.98.1
- Python 3.13.15
- NumPy 2.5.3
- SciPy 1.18.1
- one BLAS/OpenMP thread
- seed `20260916`
- 1,400 structured frames per process
- first 200 frames treated as warm-up
- 1,200 measured steps
- 12 latent entities, 16 features, 5 visible detections per frame
- local structural drift injected halfway through the stream
- relation observations and intervention-sensitive causal outcomes included
- three fresh processes per execution mode

Frame generation and materialization occur before the timed loop. For the native mode, the timed region **does include the Python → PyO3 argument conversion and the Rust → Python readout conversion**, so the measurement is for the public Python research interface rather than an internal Rust-only microbenchmark.

## Results

The table reports the median across the three independent process runs.

| Mode | Mean step | p50 | Peak RSS |
|---|---:|---:|---:|
| Frozen Python | 10,315.3 µs | 9,863.6 µs | 87.43 MiB |
| Hybrid | 10,141.9 µs | 9,719.4 µs | 86.93 MiB |
| Fully native | **78.66 µs** | **72.63 µs** | **21.99 MiB** |

For the fully native mode, the median tail latencies were:

- p95: **101.64 µs**
- p99: **129.32 µs**

For the frozen Python mode:

- p95: 12,572.93 µs
- p99: 12,917.98 µs

On this workload, the median-of-runs ratios are therefore approximately:

- **131.1× lower mean step time** versus the frozen full Python stack;
- **128.9× lower mean step time** versus the hybrid stack;
- **135.8× lower p50 latency** versus frozen Python;
- **123.7× lower p95 latency** versus frozen Python;
- **99.9× lower p99 latency** versus frozen Python;
- **4.0× lower process peak RSS** versus frozen Python.

## Raw process runs

### Frozen Python

| repetition | mean µs | p50 µs | p95 µs | p99 µs | peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 10,315.253 | 9,912.367 | 12,572.928 | 12,938.097 | 88.586 |
| 2 | 10,338.599 | 9,863.553 | 12,603.907 | 12,917.977 | 87.426 |
| 3 | 10,013.452 | 9,618.671 | 12,186.426 | 12,461.788 | 87.266 |

### Hybrid

| repetition | mean µs | p50 µs | p95 µs | p99 µs | peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 9,807.428 | 9,405.488 | 11,970.685 | 12,164.583 | 86.934 |
| 2 | 10,141.921 | 9,719.382 | 12,397.113 | 12,621.799 | 86.789 |
| 3 | 10,186.732 | 9,740.773 | 12,432.399 | 12,655.243 | 86.953 |

### Fully native

| repetition | mean µs | p50 µs | p95 µs | p99 µs | peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 77.013 | 72.488 | 97.775 | 129.320 | 22.098 |
| 2 | 81.454 | 77.304 | 101.644 | 107.828 | 21.926 |
| 3 | 78.660 | 72.628 | 102.367 | 151.211 | 21.992 |

The source run is GitHub Actions run `35091587114`; its uploaded raw text artifact is named `cortex-full-runtime-benchmark`.

## Interpretation

The hybrid result is particularly informative. Moving only the continual/plasticity controller to Rust barely changes full-frame latency, even though earlier isolated continual-stream benchmarks measured the v0.9.x controller in tens of microseconds. This demonstrates that, for this structured workload, the dominant Python cost was upstream articulation / binding / relation-memory orchestration rather than the continual controller itself.

Once the complete stateful transition is native, that interpreter/object/allocation overhead largely disappears. The result supports the architectural decision to keep experiment orchestration in Python while moving **stateful reasoning transitions** behind one coarse FFI boundary.

The RSS comparison also includes runtime/library differences: the Python and hybrid processes load NumPy/SciPy and the frozen Python stack, whereas the native process uses the much smaller native execution path. It should therefore be read as a deployment-footprint measurement, not as the byte size of Cortex state alone.

## What this does not establish

This campaign does not establish that Cortex is always ~131× faster in Rust. The ratio can change with:

- entity count and visibility;
- feature dimensionality;
- graph density;
- tensor duty cycle;
- number of stored regimes;
- split/merge activity;
- hardware and allocator behavior;
- whether the caller is Python, native Rust, or another FFI client.

The next performance work should therefore be a controlled scaling campaign across declared resource axes, with p50/p95/p99 latency, memory, and semantic correctness measured together. Optimization should target only bottlenecks that appear in those profiles.
