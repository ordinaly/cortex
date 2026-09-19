# Cortex project milestones

Status: **active roadmap**.

This document separates the next Cortex milestones into three parallel tracks:

1. **Reasoning capability** — what kinds of structural reasoning Cortex can actually perform;
2. **Engineering and specification** — which mechanisms become efficient, frozen, and promotable;
3. **External validation** — whether the measured strengths survive fair comparison and less curated data.

The tracks should remain distinct. A successful research prototype is not automatically production semantics, and a runtime optimization is not evidence of a new reasoning capability.

---

## Current coordinates

- software/runtime: `v1.0.0-rc.2`;
- frozen executable specification: `spec-v0.9.5`;
- predictive-geometry research frontier: `v1.14-R`;
- Reasoning Map completed axes:
  - recursive typed composition;
  - candidate-law induction;
  - bounded law-form discovery;
  - cross-world law transfer;
  - structural counterfactual reasoning;
  - belief revision under contradiction;
  - structural analogy / isomorphism.

The next Reasoning Map axis is **latent concept induction**.

---

# Track A — Cortex Reasoning Map

## A1. Structural analogy / isomorphism v1 — passed

### Question

Can Cortex recognize and reuse the same relational organization when both entity labels and relation labels change?

The benchmark should distinguish:

$$
\text{same surface symbols}
$$

from

$$
\text{same structural organization}.
$$

### Core design

Use paired source/target worlds with:

- independently permuted entity labels;
- independently permuted relation labels;
- partial target evidence;
- held-out relational consequences;
- graph families with both unique and non-unique correspondences.

Cortex should infer a correspondence

$$
\phi:G_s\rightarrow G_t
$$

only to the resolution justified by evidence.

### Essential controls

- **near-isomorphic negative control:** one structural edge/type is changed;
- **broken transfer control:** a source rule is invalid in the target;
- **automorphism ambiguity:** several mappings are equally valid;
- **no-label-leakage:** source/target names and token identities are disjoint;
- **scratch baseline:** target reasoning with no transferred source structure.

### Primary metrics

- correspondence precision and coverage;
- held-out consequence accuracy;
- transfer sample-efficiency gain;
- negative-transfer rate;
- unresolved rate on genuinely ambiguous mappings;
- mapping confidence/calibration;
- structural provenance.

### Important baseline

An exact graph-isomorphism algorithm should be included as an **oracle/algorithmic reference**, not treated as a competitor Cortex must beat. The scientific question is whether Cortex can discover and exploit correspondences under partial evidence and uncertainty.

### Passing claim boundary

The official campaign passed every frozen gate. With only 12 of 90 target facts observed, mean held-out coverage was **0.9718** at **1.000 resolved accuracy**; at 24 and 36 facts coverage reached **1.000**. Symmetric worlds preserved genuine automorphism ambiguity, while near-isomorphic and explicitly broken analogies were rejected in every case.

See [STRUCTURAL_ANALOGY_V1.md](STRUCTURAL_ANALOGY_V1.md).

The result supports structural analogy under declared finite relational fixtures. It does not establish unrestricted analogy or semantic understanding.

---

## A2. Latent concept induction v1

### Question

Can Cortex introduce a new latent category because doing so compresses predictive structure?

Instead of supplying every useful concept, construct observations where several entities become easier to explain under a shared latent articulation:

$$
\{e_1,e_3,e_7\}\rightarrow Z.
$$

### Required behavior

Cortex should:

- propose the latent grouping only when it improves predictive compression;
- avoid inventing a category under random/noisy controls;
- split the category when future evidence makes it insufficient;
- merge/retract it when the distinction ceases to matter.

### Primary metrics

- latent-group recovery;
- predictive gain;
- complexity cost;
- false-concept rate;
- split/merge reversibility;
- sample efficiency;
- unresolved behavior.

This would be one of the strongest tests of Cortex's core principle that structure must be earned by evidence.

---

## A3. Noisy reasoning and confidence calibration

The first counterfactual and belief-revision campaigns deliberately used deterministic evidence to isolate reasoning operations.

The next robustness campaign should add:

- finite-sample noise;
- near-threshold causal effects;
- intermittent contradiction;
- changing evidence reliability;
- missing observations;
- simultaneous belief changes.

### Primary metrics

- Brier score;
- expected calibration error;
- abstention precision;
- false-certainty rate;
- revision/recovery latency distributions;
- unaffected-structure preservation.

The goal is to determine whether `unresolved` activates **before** errors dominate.

---

## A4. Explicit proof dependency v1

Move from carrying one valid path to maintaining a proof/dependency DAG.

Test:

- multiple independent proofs;
- removal of one premise;
- proof invalidation;
- redundant support;
- contradictory proof paths;
- shortest versus merely valid derivations.

A conclusion should survive removal of one proof when another independent proof remains.

---

## A5. Planning and replanning v1

Give Cortex an explicit action model:

$$
s_t\xrightarrow{a_t}s_{t+1}
$$

and a goal state.

Test:

- multi-step plans;
- dead ends;
- reversible actions;
- action costs;
- changing transition rules;
- partial replanning after local model revision.

The Cortex-specific question is whether a local rule change causes a local plan repair rather than global relearning.

---

## A6. Open-world novelty attribution v1

Present evidence that cannot be explained by the current ontology.

Cortex must distinguish among:

- noise;
- new instance;
- new entity type;
- new relation;
- new regime.

The key metric is not merely novelty detection but **novelty attribution**: what kind of model extension was required?

---

# Track B — Engineering and specification

## B1. v1.14-R exact perceptual fast path — passed

The exact fast path passed every frozen gate.

At 288 anchors:

- perceptual computation: **48.26× faster**;
- complete fused step: **1.96× faster**;
- perceptual stage share: **2.20%**, down from about 48.9%;
- structural disagreements: **0**.

Numerical differences remained at floating-point scale. See [EXACT_PERCEPTUAL_FAST_PATH_V1_14_RESULT.md](EXACT_PERCEPTUAL_FAST_PATH_V1_14_RESULT.md).

---

## B2. v1.15-R — cache-refresh attribution / decomposition

v1.14-R re-profiling identifies **cache refresh** as the new dominant stage at about **45.5%** of the 288-anchor step. Candidate readout is second at about 21%.

The next milestone should measure cache-refresh internals before changing them:

- semantic-feature recomputation;
- outcome-probability refresh;
- dirty-anchor detection;
- pairwise distance-row updates;
- kernel-row regeneration;
- writes/bookkeeping and unattributed remainder.

No substage winner is preregistered. The attribution campaign chooses the next exact optimization target.

---

## B3. Freeze the next executable research specification

Once the selected reasoning and predictive-geometry mechanisms stabilize, create a **new frozen specification** rather than modifying `spec-v0.9.5`.

The freeze should include:

- exact public state semantics;
- confidence/unresolved semantics;
- accepted reasoning primitives;
- deterministic fixtures;
- differential test vectors;
- versioned protocol dependencies.

This is the boundary between successful research and promotable architecture.

---

## B4. Promote accepted mechanisms to Rust

Only mechanisms included in the new frozen specification should move into the native runtime.

Promotion requires:

- Python/reference oracle;
- differential Rust parity;
- state fingerprint checks;
- workload performance campaign;
- no silent semantic changes.

---

# Track C — Comparative and external validation

## C1. Reasoning profile comparison

Once A1–A3 stabilize, run the same information budgets against appropriate baselines.

Depending on the task:

- direct memory;
- Bayesian structural models;
- symbolic closure;
- graph-isomorphism algorithms;
- SAT/SMT or theorem provers;
- GNN/MPNN models;
- Transformer baselines.

Do **not** collapse these into one overall score.

The output should be a profile over:

- accuracy;
- coverage;
- calibration;
- sample efficiency;
- extrapolation depth;
- revision;
- transfer;
- compute;
- memory;
- provenance.

---

## C2. Untouched external structured-data validation

Freeze all configuration before selecting/evaluating the final external dataset.

Prefer chronological or naturally incremental data where Cortex's persistent-state design is actually relevant.

Minimum requirements:

- no benchmark-specific post-hoc tuning;
- fair online/continual baselines;
- explicit compute and memory accounting;
- preserved failed hypotheses;
- claims restricted to the observed domain.

This remains one of the strongest gates before broad scientific claims.

---

## C3. Hybrid Cortex Stage II

Only after frozen-encoder Stage I demonstrates value beyond simpler matching should Cortex feedback influence representation learning.

Questions for Stage II:

- can structural uncertainty guide representation refinement?
- can persistent entity/relation evidence improve the encoder?
- does feedback cause instability or representation collapse?
- can gains be attributed to Cortex rather than extra training?

---

# Recommended execution order

Two tracks can proceed in parallel without contaminating each other:

### Scientific reasoning track

$$
\boxed{
A2\ \text{latent concepts}
\rightarrow
A3\ \text{noise/calibration}
\rightarrow
A4\ \text{proof dependency}
}
$$

Planning and open-world novelty follow once those primitives are better characterized.

### Engineering track

$$
\boxed{
B2\ \text{cache-refresh attribution}
\rightarrow
B3\ \text{freeze spec}
\rightarrow
B4\ \text{Rust promotion}
}
$$

### Validation track

Begin C1 after analogy/noise characterization is stable. External validation C2 should precede any substantially broader claims. Hybrid Stage II follows only after the frozen-encoder comparison is convincing.

---

# Near-term definition of success

The next major scientific checkpoint should not be another perfect synthetic score.

A more meaningful checkpoint is:

> Cortex demonstrates **analogy, latent abstraction, and calibrated revision under noisy evidence**, while preserving conservative abstention and showing measurable sample-efficiency advantages over scratch reasoning.

At that point we would have evidence spanning:

$$
\text{composition}
\rightarrow
\text{law induction}
\rightarrow
\text{law-form discovery}
\rightarrow
\text{cross-world transfer}
\rightarrow
\text{counterfactual modification}
\rightarrow
\text{belief revision}
\rightarrow
\text{analogy}
\rightarrow
\text{latent abstraction}.
$$

That would justify evaluating Cortex as a broader **general structural reasoning architecture** on external data, while still avoiding claims of universal reasoning.
