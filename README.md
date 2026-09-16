# Cortex

**Cortex** is an experimental architecture for **continual structural reasoning**.

In simple terms, Cortex watches a stream of structured observations, builds a compact model of the recurring situations it encounters, and keeps that model up to date without relearning everything from scratch whenever the world changes.

It is not a chatbot, a language model, or a raw perception system. Cortex sits one level above perception: it assumes observations have already been turned into useful features, entities, events, or relations, then reasons about how those structures persist, change, return, split apart, merge again, or become genuinely novel.

**Author and project creator:** Arnold von Bauer-Gauss

## What Cortex is trying to solve

Many learning systems work well when the task and data distribution stay mostly fixed. Real environments are messier. The same situations can return after a long absence, known situations can drift gradually, one apparent concept can turn out to contain several distinct cases, and genuinely new situations can appear.

A system that reacts to every change by creating a new model will eventually fill its memory with duplicates. A system that never creates new structure becomes rigid. Cortex is an attempt to find a useful middle ground.

Its guiding principle is:

> **Explain new evidence with the smallest structural change that future evidence can justify.**

When something changes, Cortex tries increasingly expensive explanations instead of immediately assuming that the world contains a brand-new concept. It can reactivate a known situation, adapt an existing one, refine one concept into predictively distinct substructures, merge distinctions that stop mattering, or create genuinely new structure when the existing model is no longer sufficient.

This makes structural complexity reversible rather than a one-way accumulation of memories.

## A small example

Imagine Cortex is observing a machine over several years.

At first it learns a normal operating regime. Later the machine starts behaving differently in winter. Cortex may discover that this is not a new machine state at all, but a recurring seasonal regime it has seen before. Months later, one operating regime may slowly drift as parts wear down; Cortex can adapt the existing concept instead of inventing another one. If that regime eventually separates into two reliably different patterns, Cortex can test a possible split prospectively and only accept it if future observations show that the distinction improves prediction.

If the two patterns later become indistinguishable again, Cortex can merge them.

The goal is therefore not merely to predict the next observation. The goal is to maintain a **small, useful, revisable internal structure** for an environment that changes over time.

## How Cortex reasons

The current native runtime follows a layered transition:

```text
structured observations
        ↓
entity identity + nuisance-transform handling + binding
        ↓
sparse relation and intervention-sensitive causal evidence
        ↓
12-dimensional public articulation summary
        ├──────────────→ bounded fuzzy historical memory
        └──────────────→ continual regime reasoning and plasticity
```

The two downstream paths deliberately receive the same public articulation summary. Fuzzy-memory reconstruction is not fed back into the continual controller. This preserves the semantics of the frozen Python research stack while keeping approximate historical compression separate from active reasoning.

At the continual level, Cortex distinguishes several kinds of plasticity: recurrence/reactivation, local parametric adaptation, structural splitting, later reconciliation/merging, and genuine novelty. Small residual tensors can detect structured predictive error inside an otherwise stable concept, but those tensor computations are conditionally activated rather than being paid for continuously.

## What makes Cortex different

Cortex is built around bounded, explicit state rather than an ever-growing history. It keeps finite concept and memory budgets, tracks prediction evidence and structural confidence, and exposes unresolved or budget-pressure states instead of silently pretending that capacity is unlimited.

Structural decisions are also evidence-gated. A possible split is only a proposal until future observations justify the additional complexity. A previously useful split can later be merged. Known recurrence is preferred over rediscovery. The intended behavior is therefore conservative with structure but plastic when evidence becomes persistent and predictive.

The architecture also separates exact active reasoning from approximate historical memory. Compression is allowed only through declared distortion budgets; exhausting a budget does not silently authorize a lossy structural edit.

The project has been tested on controlled continual-learning fixtures, chronological real-world streams, structured synthetic worlds, and cross-language differential replays. These are empirical and engineering validation results, not evidence that Cortex is a universal predictor or a complete AGI system.

## What Cortex does not do

Cortex v1.0 is intentionally scoped as a **continual structural reasoning architecture over structured observations**. It does not currently learn vision, audio, or language representations directly from raw data. A future system can place a neural or other perceptual encoder in front of Cortex:

```text
raw data -> perceptual encoder -> structured observations -> Cortex
```

Cortex also does not assume clean task boundaries. Changes may be gradual, recurrent, ambiguous, compositional, or previously unseen.

## Native Rust runtime + Python research interface

The v1.0 architecture now has a **fully composed native stateful runtime**.

Rust owns the state transitions that occur on every frame. Python remains the research layer for datasets, experiments, baseline comparisons, plotting, notebooks, and analysis. The public boundary is deliberately coarse-grained: one `CortexRuntime.step(...)` call crosses into Rust once and executes the complete reasoning transition there.

The native workspace is divided by responsibility:

```text
crates/
  cortex-articulation/   entity identity, invariance, binding, residual/noise state
  cortex-graph/          sparse relation, causal and dependency state
  cortex-memory/         bounded Fuzzy Accordion Memory and certified recompression
  cortex-core/           continual recurrence, tensor plasticity, split/merge/novelty
  cortex-runtime/        full-frame composition of the native layers
  cortex-python/         thin PyO3 research interface
```

The lower-level Python-facing `Articulation`, `GraphEvidence`, `FuzzyMemory`, and `Cortex` objects remain available for research and debugging. `CortexRuntime` is the complete composed execution path.

## Current project status — v1.0-RC1

The migration from the frozen Python research implementation to Rust is now **functionally complete at the stateful runtime level**.

The frozen Python v0.9.5 tree remains in `reference/v0_9_5/` as an executable specification. Native modules are accepted only after deterministic differential tests reproduce the relevant Python behavior. The current composed runtime has passed end-to-end replay covering entity binding, nuisance transformations, subgroup inference, relation and causal state changes, fuzzy memory, continual prediction, recurrence/plasticity decisions, and final structural state.

CI currently verifies canonical Rust formatting, strict Clippy, release-mode Rust tests, the PyO3 extension, frozen-reference integrity, Python 3.11 and 3.13 differential suites, and a locked Maturin wheel build.

The project has also removed the old dense relation/causal preallocation from the native runtime. Relation and directed causal cells are now created lazily when evidence actually appears, changing the default storage model from an up-front quadratic carrier to sparse represented state. The worst case can still become quadratic if the environment itself produces evidence for every possible pair.

The canonical release-candidate configuration is `configs/v1_rc1.json`.

For the detailed migration record, see [`docs/NATIVE_MIGRATION.md`](docs/NATIVE_MIGRATION.md).

## First full-stack native performance result

After completing the native composition, Cortex was benchmarked in three fresh-process modes over the same deterministic 1,400-frame structured workload: frozen Python, a hybrid Python/Rust path, and the fully native `CortexRuntime` path.

Median results across three runs were:

| execution mode | mean step | p50 latency | peak process RSS |
|---|---:|---:|---:|
| Frozen Python | 10,315.3 µs | 9,863.6 µs | 87.43 MiB |
| Hybrid | 10,141.9 µs | 9,719.4 µs | 86.93 MiB |
| Fully native | **78.66 µs** | **72.63 µs** | **21.99 MiB** |

On this particular controlled workload, the fully native path was about **131× faster by mean step time** and used about **4× less peak process memory** than the frozen Python full stack. The native median p95 and p99 latencies were approximately 101.6 µs and 129.3 µs respectively.

All three modes finished with the same final prediction/state fingerprint. The large difference appeared only after the complete stateful transition moved behind one Rust boundary; migrating the continual controller alone barely changed full-frame latency. This indicates that, for this workload, Python object/allocation/orchestration overhead in articulation, binding, relation processing, and memory dominated the full Python execution cost.

These measurements are **workload-specific performance characterization, not a universal speedup claim**. Hardware, graph density, feature dimension, entity count, tensor activity, regime count, and caller language can materially change the result.

Full methodology, raw repetitions, caveats, and interpretation are in [`docs/FULL_RUNTIME_BENCHMARK.md`](docs/FULL_RUNTIME_BENCHMARK.md).

## Native scaling envelope — first optimization pass

The first native scaling campaign is complete. It swept 25 configurations across entity count, visible entities per frame, feature dimension, co-visibility topology, regime-memory budget, and tensor/refinement duty cycle, with three repetitions per configuration.

That campaign exposed an accidental hot-path cost: the 12-dimensional articulation summary was materializing and deterministically sorting the entire accumulated sparse relation/causal graph on every frame even though it needed only three graph-state counts. The runtime now maintains those counts incrementally through a lightweight `EvidenceSummary`; full graph snapshots remain available for inspection, serialization, and differential checking, but are no longer required by the per-frame summary path.

Re-running the exact same 25-case campaign produced a median **31.96% reduction in mean step time** across cases. The effect grows sharply when the represented graph is large:

| scaling case | before | after | speedup |
|---|---:|---:|---:|
| 64 nominal entities | 528.97 µs | 196.77 µs | 2.69× |
| 96 nominal entities | 1,006.25 µs | 264.35 µs | 3.81× |
| 8 visible entities | 528.82 µs | 202.98 µs | 2.61× |
| dense co-visibility, 1 cluster | 528.72 µs | 196.02 µs | 2.70× |

The topology sweep is especially diagnostic. Before the fix, changing from one dense co-visibility cluster to eight sparse clusters reduced mean step time from about 529 µs to 259 µs because fewer historical graph cells had to be snapshotted and sorted. After the fix, the same sweep is approximately 196–219 µs. Accumulated represented-graph size is therefore no longer the dominant per-frame cost.

The A/B runs preserve the same structural counters, graph occupancy, memory/regime state, tensor activity, and differential-test behavior; the optimization removes redundant work rather than changing Cortex semantics. The committed medians are in [`benchmarks/results/native_scaling_summary_optimized_medians.csv`](benchmarks/results/native_scaling_summary_optimized_medians.csv), with the direct before/after comparison in [`benchmarks/results/native_scaling_summary_ab.csv`](benchmarks/results/native_scaling_summary_ab.csv).

The next profiling target is now narrower. Remaining scaling is driven primarily by work tied to the current frame—especially visible-pair processing and feature-dimensional articulation—rather than by the total historical graph size. Before introducing SIMD, parallelism, alternative sparse structures, or low-rank approximations, the next benchmark pass should attribute frame time across articulation/binding, graph observation, public-summary construction, fuzzy memory, and continual plasticity. Regime-budget stress also needs a separate fixture that actually realizes enough regimes to create budget pressure; the current sweep usually realizes only two or three.

After the native resource envelope is characterized further, the main scientific gate remains untouched external structured-data validation with frozen configuration and fair baselines.

## Reference freeze

The exact files frozen from v0.9.5 are hashed in:

```text
reference/v0_9_5/SHA256SUMS.txt
```

Verify them with:

```bash
pytest tests/test_freeze.py
```

Files in `reference/v0_9_5/` should not be edited. A semantic change to the executable specification requires a new reference version.

## Differential migration contract

The Rust implementation is not considered equivalent merely because it compiles or produces similar aggregate benchmark scores.

Differential gates replay the same chronological observations through the frozen Python implementation and the native implementation, then compare numerical predictions and state as well as discrete structural decisions. Floating-point quantities use explicit tolerances; structural decisions are expected to agree.

This gives Cortex two complementary roles:

```text
Python v0.9.5     = readable executable specification
Rust v1.0-RC1     = native stateful reasoning implementation
Python package    = research and experiment interface
```

Future low-level optimizations can therefore be checked against the same behavioral oracle instead of silently changing the algorithm while improving performance.

## Build and test

Prerequisites:

- Rust 1.98.1, pinned in `rust-toolchain.toml`
- Python 3.11+
- Maturin 1.15.0

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install "maturin==1.15.0" pytest numpy scipy
python -m pip install -e .
pytest
```

For the Rust workspace directly:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test --workspace --release
```

## Research maturity and claims

Cortex is still an experimental research system. v1.0-RC1 means the architecture has a reproducible native implementation and increasingly explicit behavioral contracts; it does **not** mean that every research question around the architecture is closed.

The strongest current claims are about the implemented contracts: bounded state, explicit structural plasticity, reversible refinement, sparse native relation/causal storage, cross-language behavioral parity on the frozen migration fixtures, and the measured performance of the published benchmark workloads.

Open work includes continued native scaling attribution after the first optimization pass, long-duration resource stress, regime-budget saturation fixtures, untouched external full-stack structured benchmarks, broader baseline comparison, and formal results for selected consistency/boundedness properties.

## License and attribution

Cortex is licensed under the **Apache License, Version 2.0** (`Apache-2.0`).

Original author and project creator: **Arnold von Bauer-Gauss**.

The repository includes a top-level `NOTICE` file containing the project attribution. Under Section 4(d) of Apache-2.0, redistributions of derivative works that include this work must preserve the applicable attribution notices from `NOTICE` in a readable form. See `LICENSE`, `NOTICE`, and `AUTHORS.md`.

This attribution requirement does **not** imply endorsement of derivative products by the author.
