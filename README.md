# Cortex

**Cortex** is an experimental architecture for **resource-bounded continual structural reasoning**.

Cortex consumes structured observations and maintains a compact, revisable model of entities, relations, regimes, recurrence, local change, structural splits, later reconciliation, and genuine novelty. It is designed around a simple principle:

> **Explain new evidence with the smallest structural change that future evidence can justify.**

Cortex is not a chatbot, foundation model, or raw perception system. It can sit behind a perceptual encoder, but its own role is persistent structural reasoning over time.

## Current status

Cortex currently has three independent provenance coordinates:

| Surface | Current state |
|---|---|
| software/runtime | **v1.0.0-rc.2** |
| frozen executable specification | **spec-v0.9.5** |
| research frontier | **v1.13-R** |

The native runtime migration is complete at the composed stateful level. The frozen Python specification remains an executable oracle for differential validation. The research frontier is intentionally separate: experimental mechanisms are not promoted into the native contract merely because they look promising.

For a detailed explanation of how Cortex works, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For the live project summary, see [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md); the consolidated milestone ladder is [docs/PROJECT_MILESTONES.md](docs/PROJECT_MILESTONES.md).

## Architecture

The canonical detailed architecture guide is [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The composed native path is:

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

Rust owns the stateful hot path:

```text
crates/
  cortex-articulation/   entity identity, invariance, binding, residual/noise state
  cortex-graph/          sparse relation, causal and dependency state
  cortex-memory/         bounded fuzzy historical memory
  cortex-core/           recurrence, adaptation, split/merge/novelty
  cortex-runtime/        complete composed frame transition
  cortex-python/         thin PyO3 research interface
```

Python remains the experiment layer for datasets, benchmark orchestration, analysis, plotting, and research prototypes.

## Behavioral contract

Cortex is explicitly bounded and structurally conservative:

- known recurrence should be reactivated rather than rediscovered;
- local adaptation is preferred to unnecessary new structure;
- structural splits require prospective predictive evidence;
- distinctions can later merge when they cease to matter;
- bounded memory cannot silently rewrite active structural state;
- capacity exhaustion produces explicit `budget_pressure` / `unresolved` state;
- structural actions remain inspectable;
- approximate historical memory is kept separate from active reasoning.

The release-candidate contract is recorded in [docs/V1_CONTRACT.md](docs/V1_CONTRACT.md).

## Native implementation and validation

The frozen Python tree under `reference/v0_9_5/` is protected by SHA-256 hashes and is intentionally historical and immutable.

Native layers are accepted through differential replay rather than by compilation alone. CI checks include Rust formatting and Clippy, release-mode Rust tests, PyO3 build, Python differential suites, frozen-reference integrity, wheel packaging, and version consistency.

See:

- [docs/NATIVE_MIGRATION.md](docs/NATIVE_MIGRATION.md)
- [docs/VERSIONING.md](docs/VERSIONING.md)

## Performance results

### Full-stack native migration

On the controlled 1,400-frame full-stack workload:

| execution mode | mean step | p50 latency | peak RSS |
|---|---:|---:|---:|
| Frozen Python | 10,315.3 µs | 9,863.6 µs | 87.43 MiB |
| Hybrid | 10,141.9 µs | 9,719.4 µs | 86.93 MiB |
| Fully native | **78.66 µs** | **72.63 µs** | **21.99 MiB** |

On that specific workload, the fully native path was about **131× faster by mean step time** and used about **4× less peak process memory** than the frozen Python stack. These are workload-specific measurements, not universal speedup claims.

See [docs/FULL_RUNTIME_BENCHMARK.md](docs/FULL_RUNTIME_BENCHMARK.md).

### Native scaling and attribution

A first native scaling sweep exposed an accidental global graph-snapshot cost. Replacing that hot-path materialization with incremental evidence counts reduced median mean-step time by **31.96% across the 25-case campaign**, with substantially larger gains in graph-heavy cases.

Top-level stage attribution then showed that native **articulation** accounts for roughly **83–95%** of attributed runtime in the tested stress cases. Fine-grained attribution localized that cost further: cyclic transform-distance evaluation accounts for roughly **85–96%** of attributed articulation work.

See:

- [docs/NATIVE_SCALING_SWEEP_V1.md](docs/NATIVE_SCALING_SWEEP_V1.md)
- [docs/NATIVE_STAGE_ATTRIBUTION.md](docs/NATIVE_STAGE_ATTRIBUTION.md)
- [docs/ARTICULATION_ATTRIBUTION.md](docs/ARTICULATION_ATTRIBUTION.md)

## Research frontier — v1.13-R

The active research line studies how Cortex should allocate predictive distinctions rather than assuming every perceptual distinction deserves permanent structure.

The progression has established:

- fuzzy predictive geometry;
- prospective multi-hop relational inference on previously unseen pairs;
- predictive rate-distortion resolution;
- hysteresis to prevent unstable resolution switching;
- local rather than global adaptive resolution;
- exact shared-state caching;
- sparse geometry/decision refresh;
- the empirical sparse-maintenance crossover;
- exact one-pass prequential fusion;
- retained-mass support tradeoffs;
- internal fused-stage attribution.

The latest research profiler finds **perceptual responsibility computation** to be the dominant fused-stage cost at 72, 144, and 288 anchors, accounting for about **49–54%** of measured time. The next exact optimization target is therefore a field-specific perceptual fast path that reuses the invariant that stored anchors are already normalized.

The concise research record is in [docs/RESEARCH_HISTORY.md](docs/RESEARCH_HISTORY.md). The latest profiling report is [docs/FUSED_STAGE_ATTRIBUTION_V1.md](docs/FUSED_STAGE_ATTRIBUTION_V1.md).

## Relational reasoning

Cortex already has a research demonstration of prospective multi-hop relational inference: previously unseen pairs can be ranked from relational paths without direct exposure. In synthetic variable-depth tests, the heat-kernel relation operator handled first supporting path depths 3, 5, and 7 without being told the required depth in advance.

This does **not** yet establish symbolic logic in the general sense.

The first **kinship-composition-v1** campaign now tests recursive typed relation composition directly. Cortex learned from solved structured paths of length at most 5, then was evaluated on unseen people, unseen noisy graphs and path lengths 6–10. Primitive relation tokens and answer labels were randomly permuted independently for every seed.

Across 20 seeds, Cortex achieved **1.000 accuracy / 1.000 coverage**. Exact sequence memorization had **0.000 coverage**, while a weak last-relation baseline achieved **0.222 accuracy**. A hand-engineered clipped-count symbolic baseline also achieved **1.000**, so this establishes compositional logical generalization in the declared fixture, not a unique Cortex advantage.

When one required composition transition was deliberately withheld from training, Cortex produced **0.000 coverage** on queries requiring that rule rather than fabricating an answer.

A follow-up **algebra-induction-v1** campaign then tested whether Cortex could recover genuinely unobserved products from global algebraic consistency. With 40% of anonymous operation tables hidden, Cortex selected among evidence-gated associativity, commutativity and identity laws, then closed the table under the supported laws. It achieved **1.000 resolved accuracy** on both a hidden cyclic group and a hidden noncommutative dihedral group, with 0.985 and 1.000 coverage respectively. On a non-associative subtraction control it activated none of the candidate laws and left all held-out products unresolved.

This is stronger than recursive rule application, but it remains **candidate-law induction** rather than unrestricted theorem discovery: the candidate equation forms are supplied in advance. See [docs/ALGEBRA_INDUCTION_V1.md](docs/ALGEBRA_INDUCTION_V1.md).

The next **algebra-form-induction-v1.7** campaign removed that named-law vocabulary. Cortex generated **375 canonical equation forms** from a bounded anonymous grammar, assigned fuzzy structural membership, separated predictive activation from discovery, and restricted each law to an empirically supported applicability basin. Across 120 hosted cases it achieved **1.000 resolved accuracy on every structured family**, discovered the rebracketing form on every cyclic and dihedral seed, rejected operand swap on every noncommutative dihedral seed, and made **zero resolved predictions on all 20 random magmas**. See [docs/ALGEBRA_FORM_INDUCTION_V1.md](docs/ALGEBRA_FORM_INDUCTION_V1.md).

The follow-up **algebra-transfer-v1.6** campaign then tested whether those equation forms survive replacement of the underlying world. On independently relabeled target algebras with different orders, transferred forms improved mean held-out coverage over scratch Cortex by **+0.320 at 20% target evidence**, **+0.486 at 30%**, and **+0.537 at 40%**, while preserving zero wrong resolved compatible predictions and complete abstention on random-magma controls. A failed v1.5 negative-transfer campaign is retained in the research record.

The first **Cortex Reasoning Map v1** campaign, **counterfactual-reasoning-v1**, moved outside algebra. Across 140 untouched causal worlds, Cortex learned intervention-sensitive direct structure, composed unseen multi-hop influence, evaluated temporary edge-removal counterfactuals, preserved redundant paths, returned unresolved on an under-evidenced causal bridge, produced valid causal paths, and restored its base-state fingerprint exactly. The direct evidence was deliberately deterministic, so this establishes the structural reasoning primitive rather than noisy real-world causal discovery. See [docs/COUNTERFACTUAL_REASONING_V1.md](docs/COUNTERFACTUAL_REASONING_V1.md) and [docs/REASONING_MAP_V1.md](docs/REASONING_MAP_V1.md).

The second Reasoning Map campaign, **belief-revision-v1**, tested cumulative contradiction and recovery on a causal belief embedded in the only route through a small graph. All 60 untouched official cases passed. Retraction latency scaled with prior support at **7 / 14 / 21** contradiction batches for 1 / 2 / 3 initial causal batches, while recovery took **3 / 5 / 7** causal batches. Cortex passed through `unresolved`, resisted a single contradictory batch, remained unresolved under ambiguous evidence, propagated belief state downstream, and preserved unrelated structural evidence exactly. See [docs/BELIEF_REVISION_V1.md](docs/BELIEF_REVISION_V1.md).

## Hybrid neural/Cortex path

Hybrid Cortex Stage I feeds frozen neural embeddings into the unchanged native reasoning runtime:

```text
raw data → frozen neural encoder → metric-calibrated embeddings → Cortex
```

The encoder remains frozen so experiments can ask whether Cortex adds persistent structural inference beyond the representation itself before introducing feedback learning.

See [docs/HYBRID_STAGE_I.md](docs/HYBRID_STAGE_I.md).

## Repository layout

```text
crates/                 native Rust runtime
python/                 Python package/research boundary
research/               active and reproducible research implementations
benchmarks/             benchmark drivers and committed aggregate results
tests/                  native/Python/differential/research tests
reference/v0_9_5/       immutable frozen executable specification
docs/                   current project and benchmark documentation
configs/                canonical runtime configuration
```

Committed benchmark summaries are intentionally retained as scientific provenance. Superseded narrative milestone documents and one-shot workflow artifacts are not kept on the active documentation surface; their full history remains available through Git.

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

python scripts/check_version_consistency.py
pytest

cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test --workspace --release
```

## Research maturity

Cortex is an experimental research system. The current evidence supports the implemented contracts and the specific published benchmark results; it does **not** establish that Cortex is a universal predictor, a general substitute for neural networks, or an AGI architecture.

The largest remaining scientific gate is **untouched external validation with frozen configuration and fair baselines**, including direct tests of compositional relational generalization.

## License and attribution

Cortex is licensed under the **Apache License, Version 2.0**.

Original author and project creator: **Arnold von Bauer-Gauss**.

See `LICENSE`, `NOTICE`, and `AUTHORS.md`.
