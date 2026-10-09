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
| predictive-geometry research frontier | **v1.22-R** |
| autonomy/plasticity research default | **Relational Regime Plasticity v0.1** |

The stable runtime and the research frontiers are intentionally separate. The frozen executable specification remains the behavioral oracle, while experimental mechanisms advance independently until they have earned promotion through explicit validation. Predictive-geometry engineering remains at **v1.22-R**; the autonomous relational-learning track has frozen **Relational Regime Plasticity v0.1** as its current research default. **Autonomy Robustness & Temporal Plasticity v1** has completed Phase B; source-aware current evidence and execution freshness passed their declared research gates.

For a detailed explanation of how Cortex works, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For the live project summary, see [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md); the consolidated milestone ladder is [docs/PROJECT_MILESTONES.md](docs/PROJECT_MILESTONES.md).

## Architecture

The canonical detailed architecture guide is [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The core information flow is:

```text
structured observations
        ↓
entity identity + nuisance handling + binding
        ↓
sparse relational / causal structure
        ↓
compact public articulation state
        ├──────────────→ bounded historical memory
        └──────────────→ continual structural reasoning
```

The implementation is split between a stable runtime and an experimental research layer. Implementation details are documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); they are secondary to the research questions summarized here.

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

## Implementation and validation

The stable runtime is differentially checked against the frozen executable
specification where their contracts overlap. Default pull-request CI protects
the frozen reference, repository portability, version consistency, packaging,
and the active correctness fixtures. Full research campaigns are opt-in and run
through the dedicated research-benchmark workflow rather than on every change.

Versioning and validation conventions are documented in
[docs/VERSIONING.md](docs/VERSIONING.md). Historical implementation and
performance reports remain available under `docs/`.

## Research frontier — v1.22-R

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

Recent engineering work has focused on making the adaptive predictive-geometry path cheaper without changing its behavior. The important current results are:

- v1.19-R used low-overhead independent replay profiling and measured cache refresh at **37.22%** of the 288-anchor fused step;
- v1.20-R introduced **lazy distance materialization**, improving 288-anchor refresh by **1.132×** and the complete fused step by **1.025×** with zero numerical or structural discrepancy;
- v1.21-R repeated the same low-overhead profile after that promotion. Cache refresh remains the largest coarse stage at **35.55%**, followed by candidate readout (**21.21%**), hysteresis/state scan (**19.09%**), and complexity refresh (**13.94%**).
- v1.22-R tested a reusable 2-D single-buffer kernel refresh. It preserved exact behavior but was slower in development: at 288 anchors median refresh speedup was **0.990×** and complete fused-step speedup **0.988×**, with no fused-step wins across six seeds. The candidate was not promoted and the official seeds were not run.

The v1.22-R allocation-elision hypothesis is therefore closed as a development-stage negative result. If performance engineering continues, the next candidate should come from a different coarse stage or a genuinely different cache algorithm rather than another layout-only rewrite. Candidate readout remains the next-largest measured stage at about **21%**.

See [docs/SINGLE_BUFFER_KERNEL_REFRESH_V1_22_RESULT.md](docs/SINGLE_BUFFER_KERNEL_REFRESH_V1_22_RESULT.md), [docs/POST_LAZY_STAGE_PROFILE_V1_21_RESULT.md](docs/POST_LAZY_STAGE_PROFILE_V1_21_RESULT.md), and [docs/RESEARCH_HISTORY.md](docs/RESEARCH_HISTORY.md) for the detailed engineering history, including retained negative results.

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

The third Reasoning Map campaign, **structural-analogy-v1**, independently renamed both entities and relation types across paired worlds. Cortex maintained a version space of consistent correspondences and resolved only invariant consequences shared by every survivor. On untouched seeds, held-out resolved accuracy was **1.000**; with only 12 of 90 target facts, mean coverage was **0.9718**, reaching **1.000** at 24 facts. Symmetric automorphisms remained genuinely ambiguous, while near-isomorphic and explicitly broken analogies were rejected in every case. See [docs/STRUCTURAL_ANALOGY_V1.md](docs/STRUCTURAL_ANALOGY_V1.md).

The fourth Reasoning Map campaign, **latent-concept-induction-v1**, removes the
assumption that every useful category is supplied in advance. Cortex evaluated
all **203** partitions of six opaque entities under a predictive-description
objective, recovered every planted two- and three-concept structure at
**1.000 partition F1**, selected six singletons in every no-shared-concept
control, preserved all deliberately ambiguous memberships, and passed a
**create → split → merge → retract** concept lifecycle. Mean held-out gain over
flat per-entity prediction was **0.0556 / 0.0288 nats per outcome** for the two
grouped families. See
[docs/LATENT_CONCEPT_INDUCTION_V1.md](docs/LATENT_CONCEPT_INDUCTION_V1.md).

The next integration campaign, **concept-semantic-integration-v1**, feeds those
internally discovered concept IDs directly into semantic grounding. On 80
untouched official cases, adversarial neural naming improved from **0.3333**
raw accuracy to **1.000 Cortex semantic accuracy at full coverage**. Replacing
the complete label vocabulary changed no concept identities, a genuinely
unnamed concept remained unresolved at **0.3333** support, and semantic lineage
through concept split, merge, and retraction was exact. See
[docs/CONCEPT_SEMANTIC_INTEGRATION_V1.md](docs/CONCEPT_SEMANTIC_INTEGRATION_V1.md).


The fifth Reasoning Map campaign, **explicit-proof-dependency-v1**, moves from
carrying one valid reasoning path to maintaining explicit finite proof
alternatives with transitive fact/rule dependencies. Across **120 untouched
official cases**, every frozen gate passed: redundant support survived removal
of one proof, a longer proof took over when the shortest derivation was broken,
shared hidden premises invalidated all dependent branches, simultaneous
positive/negative proofs remained unresolved, nested invalidation stayed local,
and unrelated premise removal left the target explanation unchanged. A
single-proof baseline failed every alternate-support case. See
[docs/PROOF_DEPENDENCY_V1.md](docs/PROOF_DEPENDENCY_V1.md).

The sixth Reasoning Map campaign, **planning-replanning-v1**, adds a
finite deterministic action model with costs, add/delete effects, temporal
support provenance, local repair, global fallback, and explicit infeasibility.
Across **140 untouched official cases**, every frozen gate passed. On the **80
local-repair cases**, Cortex matched full uniform-cost replanning in solution
cost while expanding **2.425 states on average versus 27.5**, a mean expansion
ratio of **0.08745**. Irrelevant action-model changes left plan/support
fingerprints unchanged, and cases requiring earlier rerouting correctly fell
back to full replanning. See
[docs/PLANNING_REPLANNING_V1.md](docs/PLANNING_REPLANNING_V1.md).

The seventh Reasoning Map campaign,
**open-world-novelty-attribution-v1**, turns novelty detection into competing
structural explanations. Across **160 untouched official cases**, exact
attribution and edit locality were **1.000**, the false persistent edit rate was
**0.000**, ambiguous instance-vs-type evidence stayed unresolved in every case,
and transient excursion/recovery produced no structural rewrite. The
nearest-known-type and global-reset baselines reached **0.375** and **0.125**
exact attribution accuracy. See
[docs/OPEN_WORLD_NOVELTY_ATTRIBUTION_V1.md](docs/OPEN_WORLD_NOVELTY_ATTRIBUTION_V1.md).

The first multi-axis integration campaign,
**novelty-aware-replanning-v1**, composes novelty attribution with explicit
planning dependencies. Across **160 untouched official cases**, attribution,
revision-scope decisions, model locality, and repair optimality were all
**1.000**. Noise and irrelevant structural novelty caused zero planning search;
relevant instance/relation/type novelty triggered local repair; regime change
triggered global replanning; unresolved novelty deferred execution without
model mutation. Local repairs used a mean **0.03846×** the state expansions of
full replanning while preserving the same minimum cost. See
[docs/NOVELTY_AWARE_REPLANNING_V1.md](docs/NOVELTY_AWARE_REPLANNING_V1.md).

Within the Reasoning Map, the next scientific milestone is **Open-World Robustness v1**. The autonomous-learning **Autonomy Robustness & Temporal Plasticity v1** milestone has completed Phase B.

## Autonomous Cortex path

The primary architecture now treats Cortex as an **autonomous structural
learner with Cortex-owned persistent memory**. A neural model is not part of
the main cognitive loop.

Structured chronological datasets can be presented directly through
`cortex.autonomy.ObservationFrame` and `train_dataset(...)`. This is
continual exposure rather than gradient training: observations update Cortex
state directly, later evidence may revise earlier structure, and the resulting
snapshot can be restored without a teacher model.

Cortex also has an experimental **Logical Processing Unit (LPU) v0.1**. The LPU
is a stateless exact-computation coprocessor for temporary articulated values
and relations. It currently supports Boolean operations, equality/comparison,
exact rational arithmetic, and directed reachability with a witness path.

The architectural boundary is intentional:

```text
structured observations
        |
        v
   Cortex memory
        |
        +----> structural learning / revision / abstraction
        |
        +----> temporary articulation ----> stateless LPU
```

The LPU does not own facts, predicates, ontology, or persistent memory, and its
results are not automatically written back into Cortex. Historical
neural/Cortex experiments remain available in Git history and research reports,
but the former Ollama/Qwen gateway and hybrid runtime are no longer part of the
active package surface.

See
[docs/AUTONOMOUS_CORE_LPU_V0_1.md](docs/AUTONOMOUS_CORE_LPU_V0_1.md).

### Relational regime plasticity

The first dedicated plasticity sweep showed that cumulative relation evidence
could detect contradiction early while still requiring roughly **17.8x prior
history** to re-execute a genuine replacement relation. The accepted
dual-timescale mechanism keeps durable history exact, adds fast current-regime
evidence, and uses explicit hysteretic contradiction pressure.

Its 270-policy Pareto calibration selected a research default that switches a
clean OLD -> NEW regime in **9 observations** for every tested prior mass
4 / 8 / 16 / 32 / 64, preserves the original v0.2 contradiction-detection
latency, produces zero declared false switches under transient and stationary
noise controls, reverses in 9 observations, preserves unrelated relations, and
retains historical counts exactly.

This is a research mechanism, not a change to `spec-v0.9.5`. See
[docs/RELATIONAL_REGIME_PLASTICITY_V0_1.md](docs/RELATIONAL_REGIME_PLASTICITY_V0_1.md).

## Repository layout

```text
research/               active and reproducible research implementations
benchmarks/             benchmark drivers and committed aggregate results
tests/                  validation and research tests
reference/v0_9_5/       immutable frozen executable specification
docs/                   architecture, project status and experiment reports
crates/                 stable runtime implementation
python/                 Python package / runtime boundary
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


### Finite Sigma–Lambda oracle and SL-Bench

The opt-in [SL-Bench v0.1 research module](docs/SL_BENCH_V0_1.md) provides an exact
finite predictive oracle, adversarial streams, a bounded approximate learner,
and a binary `EvidenceAtom` prediction adapter. [Pilot results](docs/SL_BENCH_V0_1_RESULTS.md)
reject aggressive folding as a default improvement: active cells shrink, but
retained evidence does not, and adversarial prediction loss increases. The
adapter therefore defaults to no folding. This work does not change the native
runtime or establish whole-theory correctness.

The [v0.2 joint-trace extension](docs/SL_BENCH_TRACES_V0_2.md) adds bounded
multi-step action-conditioned prediction. It tests hidden temporal correlations
and delayed distinctions against the exact oracle, with an independent-noise
control to expose the cost of unnecessary detail.
