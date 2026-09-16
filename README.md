# Cortex

**Cortex** is an experimental architecture for **continual structural reasoning**.

In simple terms, Cortex is designed to watch a stream of structured observations, build a compact picture of the recurring situations it encounters, and keep that picture up to date without relearning everything from scratch every time the world changes.

It is not a chatbot, a language model, or a raw perception system. Cortex sits one level above perception: it assumes that observations have already been turned into useful features, events, entities, or relations, then tries to reason about how those structures persist, change, return, split apart, or disappear over time.

**Author and project creator:** Arnold von Bauer-Gauss

## What Cortex is trying to solve

Many learning systems work well when the task and data distribution stay mostly fixed. Real environments are messier. The same situations can return after a long absence, known situations can drift gradually, one apparent concept can turn out to contain several distinct cases, and genuinely new situations can appear.

A system that reacts to every change by creating a new model will eventually fill its memory with duplicates. A system that never creates new structure becomes rigid. Cortex is an attempt to find a middle ground.

Its guiding principle is:

> **Explain new evidence with the smallest structural change that future evidence can justify.**

When something changes, Cortex tries increasingly expensive explanations instead of immediately assuming that the world contains a brand-new concept.

For example, it can decide that:

1. an old situation has simply returned, so an existing concept should be reactivated;
2. the current concept is still correct but its parameters have shifted, so it should adapt locally;
3. one concept actually contains two predictively different substructures, so it should be refined into two children;
4. a distinction that used to matter no longer matters, so two children can be merged again;
5. nothing already known explains the evidence well enough, so a genuinely new concept should be created.

This makes structural complexity reversible rather than a one-way accumulation of memories.

## A small example

Imagine Cortex is observing a machine over several years.

At first it learns a normal operating regime. Later the machine starts behaving differently in winter. Cortex may discover that this is not a new machine state at all, but a recurring seasonal regime it has seen before. Months later, one operating regime may slowly drift as parts wear down; Cortex can adapt the existing concept instead of inventing another one. If that regime eventually separates into two reliably different patterns, Cortex can test a possible split in the background and only accept it if future observations show that the distinction improves prediction.

If the two patterns later become indistinguishable again, Cortex can merge them.

The goal is therefore not merely to predict the next observation. The goal is to maintain a **small, useful, revisable internal structure** for an environment that changes over time.

## What makes Cortex different

Cortex is built around bounded, explicit state rather than an ever-growing history. It keeps a limited library of concepts or regimes, tracks confidence and prediction statistics for them, and exposes memory pressure instead of silently pretending that capacity is unlimited.

Its plasticity is also separated into different kinds of change. Reactivating an old concept is cheaper than modifying one, modifying one is cheaper than splitting it, and splitting is cheaper than declaring something genuinely novel. This gives Cortex a way to balance stability against adaptability instead of controlling everything with one global learning rate.

Recent versions also use small residual tensors to detect whether prediction errors contain coherent internal structure. Those tensor computations are themselves conditional: Cortex can put them to sleep when a concept is stable and wake them when cheap monitors indicate that more detailed reasoning is justified.

The current research prototype has been tested on controlled continual-learning fixtures and several chronological real-world streams. These experiments are evidence about the behavior of the architecture, not a claim that Cortex is a universal predictor or a complete AGI system.

## What Cortex does not do

Cortex v1.0 is intentionally scoped as a **reasoning layer over structured observations**. It does not currently learn vision, audio, or language representations directly from raw data. A future system could place a neural or other perceptual encoder in front of Cortex:

```text
raw data -> perceptual encoder -> structured observations -> Cortex
```

Cortex also does not assume that every change has a clean task boundary. The continual-learning controller is designed to work on a stream where changes may be gradual, recurrent, ambiguous, or previously unseen.

## Why Rust and Python

The v1.0 architecture separates the project into two layers.

**Rust owns the stateful reasoning core.** It is responsible for bounded regime memory, recurrence and novelty evidence, tensor plasticity, split/merge logic, sparse relation state, and other operations that are performed on every observation.

**Python owns the research interface.** It remains responsible for loading datasets, running experiments, comparing baselines, plotting results, notebooks, and scientific analysis.

The Python API is intentionally coarse-grained: one call to `Cortex.step()` performs one complete observation update inside Rust. This avoids moving dozens of tiny operations back and forth across the Python/Rust boundary.

The previous Python implementation, Cortex v0.9.5-alpha, is frozen in this repository as an **executable specification**. The Rust implementation must reproduce its behavioral contracts before it is allowed to become the default backend.

## Current repository status

This repository is the **Cortex v1.0-RC1 Rust migration workspace**.

The frozen Python reference implementation lives under `reference/v0_9_5/`. The native reasoning engine is being migrated into `crates/cortex-core`, the sparse relation/dependency layer lives in `crates/cortex-graph`, the Python extension is implemented with PyO3 in `crates/cortex-python`, and the research-facing Python package lives under `python/cortex/`.

Golden behavioral traces under `tests/fixtures/` cover local adaptation, concept splitting, genuine novelty, and split-followed-by-merge behavior. Continuous integration is expected to replay those traces through the native implementation and compare the results against the frozen Python reference within explicit numerical tolerances.

The canonical release-candidate configuration is `configs/v1_rc1.json`.

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

## Build

Prerequisites:

- Rust 1.98.1, pinned in `rust-toolchain.toml`
- Python 3.11+
- Maturin 1.15.0

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip maturin==1.15.0 pytest numpy
pip install -e .
pytest
```

## Differential migration gate

The Rust implementation is not considered equivalent merely because it compiles or produces similar aggregate benchmark scores.

The migration gate replays the same chronological observations through the frozen Python implementation and the native implementation, then checks prediction values, current concept identity, structural events, prototype counts, memory pressure, and final structural state. Floating-point quantities use explicit tolerances; discrete structural decisions are expected to agree.

This gives the project two complementary implementations:

```text
Python v0.9.5  = readable executable specification
Rust v1.0 core = optimized production implementation
```

Future low-level optimizations can therefore be checked against the same behavioral oracle.

## Current scope of the migration

The first Rust migration targets the v0.9.5 continual-learning and plasticity hot path. Earlier Cortex work on entity formation, invariance, binding, relations, and causal structure is a separate migration phase.

This is deliberate. The project is being moved into native code behind explicit behavioral contracts rather than rewritten all at once.

## License and attribution

Cortex is licensed under the **Apache License, Version 2.0** (`Apache-2.0`).

Original author and project creator: **Arnold von Bauer-Gauss**.

The repository includes a top-level `NOTICE` file containing the project attribution. Under Section 4(d) of Apache-2.0, redistributions of derivative works that include this work must preserve the applicable attribution notices from `NOTICE` in a readable form. See `LICENSE`, `NOTICE`, and `AUTHORS.md`.

This attribution requirement does **not** imply endorsement of derivative products by the author.
