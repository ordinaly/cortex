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
- predictive-geometry research frontier: `v1.22-R`;
- Reasoning Map completed axes:
  - recursive typed composition;
  - candidate-law induction;
  - bounded law-form discovery;
  - cross-world law transfer;
  - structural counterfactual reasoning;
  - belief revision under contradiction;
  - structural analogy / isomorphism;
  - latent concept induction.

The next independent Reasoning Map axis is **explicit proof dependency**.
Before that, a near-term integration milestone should connect internally
discovered concept identities to semantic grounding and semantic revision.

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

## A2. Latent concept induction v1 — passed

### Question

Can Cortex introduce a new latent category because doing so compresses
predictive structure?

The official answer is **yes within the declared finite partition grammar**.

### Passing result

Across untouched official seeds 700–719:

- two-concept minimum partition F1: **1.000**;
- three-concept minimum partition F1: **1.000**;
- no-shared-concept control correctness: **1.000**;
- mean held-out gain over flat per-entity prediction:
  **0.0556 / 0.0288 nats per outcome**;
- minimum grouped-case gain over one global concept:
  **0.2426 nats/outcome**;
- ambiguous-member unresolved rate: **1.000**;
- create/split/merge/retract lifecycle coverage: **1.000**;
- final concept retraction correctness: **1.000**.

The deliberately wrong geometry-decoy grouping reached at most **0.3333** F1,
while Cortex recovered every planted predictive grouping at **1.000** F1.

### Architectural consequence

A latent concept is now treated as a revisable predictive articulation rather
than a supplied class label. Exact member blocks preserve persistent concept
IDs; split and merge transitions create new IDs with explicit lineage; a
concept can disappear when predictive evidence no longer justifies shared
structure.

See [LATENT_CONCEPT_INDUCTION_V1.md](LATENT_CONCEPT_INDUCTION_V1.md).

### Concept-to-semantics integration — passed

The follow-up `concept-semantic-integration-v1` now feeds Cortex-discovered
concept IDs into semantic grounding.

Across 80 untouched official cases:

- raw neural semantic accuracy: **0.3333**;
- Cortex semantic accuracy / coverage: **1.000 / 1.000**;
- label-causal-isolation rate: **1.000**;
- unnamed-concept unresolved rate: **1.000**;
- split/merge semantic-lineage correctness: **1.000 / 1.000**;
- retraction invalidation correctness: **1.000**.

The full research path is now:

$$
\text{predictive evidence}
\rightarrow
\text{latent concept}
\rightarrow
\text{semantic label}.
$$

The next semantic extension is hierarchy/inheritance plus temporal revision
across concept lineage. The next independent reasoning axis is explicit proof
dependency.

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

## B2. v1.15-R — cache-refresh attribution / decomposition — passed

The official campaign decomposed the cache-refresh bottleneck without changing cache behavior.

At 288 anchors:

- distance-row recomputation: **41.10%**;
- kernel regeneration: **35.50%**;
- writes/bookkeeping: **15.48%**;
- semantic features: **2.90%**;
- dirty detection: **1.88%**;
- outcome-probability refresh: **1.55%**;
- mean dirty-row fraction: **13.40%**.

All paired numerical discrepancies were **0** and structural disagreements were **0**.

The measured next target is therefore semantic geometry plus kernel regeneration, not feature/probability refresh.

See [CACHE_REFRESH_ATTRIBUTION_V1_15_RESULT.md](CACHE_REFRESH_ATTRIBUTION_V1_15_RESULT.md).

---

## B3. v1.16-R — exact semantic-geometry fast path — mixed official result

The first development variant used the unit-norm dot-product identity directly.
It was fast but failed the strict cached-distance parity gate near zero because
of cancellation after the square root.

The accepted implementation instead accumulates squared feature differences
into a 2-D dirty-row matrix and reuses that squared geometry for all candidate
resolution kernels.

On untouched official seeds at 288 anchors:

- cache-refresh speedup: **1.523×**;
- complete fused-step speedup: **1.174×**;
- maximum distance error: **5.55e-17**;
- maximum kernel error: **3.33e-16**;
- dirty-mask disagreements: **0**;
- structural disagreements: **0**.

The preregistered bottleneck-shift gate nevertheless failed: cache refresh
remained **36.79%** of the profiled step, above the frozen **35%** ceiling.

The exact optimization is retained, but the full milestone is not recorded as a
clean pass.

See [EXACT_SEMANTIC_GEOMETRY_FAST_PATH_V1_16_RESULT.md](EXACT_SEMANTIC_GEOMETRY_FAST_PATH_V1_16_RESULT.md).

---

## B4. v1.17-R — optimized cache re-attribution — passed

The untouched official campaign reproduced the development ordering with exact
paired cache and prediction parity.

At 288 anchors:

- kernel regeneration: **48.63%**;
- writes/bookkeeping: **24.94%**;
- streamed semantic geometry: **13.69%**;
- semantic features: **4.31%**;
- dirty detection: **2.82%**;
- outcome-probability refresh: **2.32%**.

Raw geometry substages were:

- squared-distance accumulation: **11.40%**;
- final square root: **2.21%**.

Kernel regeneration is therefore the measured next target.

See [OPTIMIZED_CACHE_REATTRIBUTION_V1_17_RESULT.md](OPTIMIZED_CACHE_REATTRIBUTION_V1_17_RESULT.md).

---

## B5. v1.18-R — exact kernel-ladder fast path — rejected for performance

The candidate retained strict numerical and structural behavior, including zero
underflow-pattern mismatches, but the untouched official speedups did not meet
the frozen performance gates.

At 288 anchors:

- kernel speedup: **1.115×** (gate >= 1.40×);
- refresh speedup: **1.017×** (gate >= 1.10×);
- fused speedup: **0.998×** (gate >= 1.02×);
- maximum kernel error: **1.89e-15**;
- structural disagreements: **0**.

The development timing overestimated the end-to-end benefit, so the candidate
is not promoted.

See [KERNEL_LADDER_FAST_PATH_V1_18_RESULT.md](KERNEL_LADDER_FAST_PATH_V1_18_RESULT.md).

---

## B6. v1.19-R — low-overhead fused-stage crossover profile — passed

The untouched official profile reproduced development with zero checksum
discrepancy.

At 288 anchors:

- cache refresh: **37.22%**;
- candidate readout: **21.80%**;
- hysteresis/state scan: **17.60%**;
- complexity refresh: **13.41%**.

The coarse stage ordering is therefore stable enough to justify another cache
experiment, while the v1.18 negative result warns against choosing that
experiment from fine-grained internal timers alone.

See [LOW_OVERHEAD_STAGE_PROFILE_V1_19_RESULT.md](LOW_OVERHEAD_STAGE_PROFILE_V1_19_RESULT.md).

---

## B7. v1.20-R — lazy distance materialization / write elision — passed

The hot predictive path uses kernels directly; the full cached distance matrix
is not required by candidate readout, local readout, or complexity evaluation.

Test a cache variant that:

- computes dirty-row squared geometry as today for exact kernel refresh;
- does not eagerly mirror dirty rows into the full symmetric distance matrix;
- marks the distance view stale;
- materializes a fully synchronized distance matrix only on explicit access;
- preserves the existing public distance API and exact diagnostic values;
- falls back to the eager path if laziness cannot be made transparent.

Officially, the lazy cache achieved **1.132× refresh speedup** and **1.025× fused-step speedup** at 288 anchors with zero numerical/state discrepancies. The public distance view remains exact on demand.

See [LAZY_DISTANCE_MATERIALIZATION_V1_20_RESULT.md](LAZY_DISTANCE_MATERIALIZATION_V1_20_RESULT.md).

---

## B8. v1.21-R — post-lazy low-overhead stage profile — passed

The untouched official profile retained cache refresh as the largest coarse stage.

At 288 anchors:

- cache refresh: **35.55%**;
- candidate readout: **21.21%**;
- hysteresis/state scan: **19.09%**;
- complexity refresh: **13.94%**.

The result preserves the v1.19 ordering while showing that cache refresh is less dominant after v1.20-R.

See [POST_LAZY_STAGE_PROFILE_V1_21_RESULT.md](POST_LAZY_STAGE_PROFILE_V1_21_RESULT.md).

---

## B9. v1.22-R — single-buffer kernel refresh / allocation elision — development rejected

The candidate replaced the temporary
`resolutions × dirty_rows × anchors` kernel block with a reusable
two-dimensional scratch buffer and evaluated one resolution at a time.

Behavioral parity was exact across the frozen development campaign:

- all prediction/state/kernel/distance errors: **0**;
- dirty-mask disagreements: **0**;
- structural disagreements: **0**;
- lazy distance behavior preserved.

Performance did not justify promotion:

- 72 anchors: refresh **0.963×**, fused step **0.989×**;
- 144 anchors: refresh **0.969×**, fused step **0.991×**;
- 288 anchors: refresh **0.990×**, fused step **0.988×**;
- 288-anchor fused wins: **0 / 6**.

The development gates therefore failed and the official seeds were deliberately
not run. The validated v1.20 lazy cache remains the default.

See [SINGLE_BUFFER_KERNEL_REFRESH_V1_22_RESULT.md](SINGLE_BUFFER_KERNEL_REFRESH_V1_22_RESULT.md).

---

## B10. Freeze the next executable research specification



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

## B11. Promote accepted mechanisms to Rust

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

### Stage II-A — semantic grounding / inference-time feedback

A research prototype is now prepared under
[hybrid-semantic-grounding-v1](HYBRID_SEMANTIC_GROUNDING_V1_PROTOCOL.md).

Its invariant is:

$$
\text{concept identity}\neq\text{current label}.
$$

Neural perception supplies candidate-label distributions. Cortex scores those
proposals jointly with relational structure and may return `reinforce`,
`reinspect`, `broaden`, or `hold` feedback. Cortex can therefore demote a
high-confidence neural label without deleting the underlying concept.

The frozen official campaign has now passed. With one adversarial swapped
pair, neural top-label accuracy of **0.600** became **0.9933 resolved accuracy**
at **0.9933 coverage** after 8 structural facts. With two swapped pairs, neural
accuracy of **0.200** became **0.960 resolved accuracy / 1.000 coverage** after
12 facts, with **0.950 correction rate** and zero false reinspection on
neural-correct concepts. Exact structural twins remained unresolved at 0.500
support.

See [HYBRID_SEMANTIC_GROUNDING_V1.md](HYBRID_SEMANTIC_GROUNDING_V1.md).

A follow-up **semantic-revision-v1** campaign now adds chronological revision
hysteresis and bounded fading evidence. Across 60 untouched official cases,
Cortex preserved the incumbent under a one-epoch contradictory neural shock,
revised every sustained two-label semantic shift through
`challenged → unresolved → revised`, recovered every original mapping, and
never revised the three unaffected concepts. Maximum shift and recovery
latencies were both **4 epochs**.

See [SEMANTIC_REVISION_V1.md](SEMANTIC_REVISION_V1.md).

The follow-up **noisy-semantic-revision-v1** campaign has also passed its frozen
official gates. Under neural dropout/errors and sparse corrupted structural
evidence, Cortex improved top-label accuracy from **0.8061 neural to 0.9639**,
reduced Brier score from **0.3563 to 0.0689**, and achieved **0.9980 trusted
accuracy** at **0.7807 coverage**. False certainty was **0.00195**, uncertainty
interception **0.9750**, shift/recovery success **1.000 / 1.000**, and
unaffected-concept preservation **1.000**.

See [NOISY_SEMANTIC_REVISION_V1.md](NOISY_SEMANTIC_REVISION_V1.md).

The grounding protocols intentionally receive concept identities as input so
they can isolate semantic adjudication and revision. The intended integration
order is therefore:

$$
\text{latent concept induction}
\rightarrow
\text{semantic grounding}
\rightarrow
\text{semantic revision}.
$$

### Stage II-B — trainable representation feedback

Only after inference-time grounding and frozen-encoder Stage I are characterized
should Cortex feedback influence neural parameters.

Questions for Stage II-B:

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
C3\text{-A}\ \text{concept/semantic integration}
\rightarrow
A4\ \text{proof dependency}
}
$$

Semantic grounding is implemented as a separable research prototype now, but
its full Cortex interpretation should consume concepts produced by A2 rather
than externally supplied concept IDs. Planning and open-world novelty follow
once those primitives are better characterized.

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
