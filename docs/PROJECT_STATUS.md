# Cortex project status

Last consolidated research frontier: **v1.20-R**  
Software/runtime: **v1.0.0-rc.2**  
Executable specification: **spec-v0.9.5**

This document is the canonical human-readable status page for the active repository. For the detailed system design, data flow, state boundaries, and reasoning mechanics, see [ARCHITECTURE.md](ARCHITECTURE.md). Historical experiment details remain reproducible from Git history, research code, and committed benchmark summaries. The prioritized cross-track roadmap is maintained in [PROJECT_MILESTONES.md](PROJECT_MILESTONES.md).

## What is stable today

### Native runtime

The composed stateful path is migrated to Rust and differential-tested against the frozen Python specification where their contracts overlap.

The native runtime includes:

- persistent entity identity and nuisance-transform handling;
- simultaneous binding;
- sparse relation and intervention-sensitive causal evidence;
- bounded historical memory;
- recurrence and local adaptation;
- prospective split promotion;
- merge/reconciliation;
- novelty;
- explicit unresolved/budget-pressure states.

The current release candidate is `v1.0.0-rc.2`.

### Research runtime

The research line has moved beyond the frozen runtime semantics. It currently studies predictive semantic geometry and adaptive representational resolution.

The current research frontier is `v1.20-R`.

v1.16-R implements an exact streamed semantic-geometry cache path selected by the v1.15-R attribution campaign. On untouched official seeds at 288 anchors, cache refresh became **1.52× faster** and the complete fused step **1.17× faster**, with maximum cached-distance error **5.55e-17**, maximum kernel error **3.33e-16**, and **0 structural disagreements**. The frozen milestone nevertheless did not pass every gate: cache refresh remained **36.79%** of the profiled step, above the preregistered **35%** bottleneck-shift ceiling.

v1.17-R then re-attributed the optimized cache. At 288 anchors, **kernel regeneration was 48.63%** of instrumented refresh, writes/bookkeeping **24.94%**, and streamed geometry **13.69%**.

v1.18-R tested an exact scale-gated kernel ladder. Numerical behavior remained effectively identical (maximum kernel error **1.89e-15**, zero underflow/dirty/structural mismatches), but the untouched official performance campaign rejected promotion: at 288 anchors the kernel path was only **1.115×** faster, refresh **1.017×**, and the complete fused step **0.998×**, below all frozen performance gates. The candidate remains research-only.

v1.19-R then re-profiled the fused research path with one timed stage per fresh replay to reduce observer interference. The untouched official profile reproduced development: at 288 anchors, cache refresh was **37.22%**, candidate readout **21.80%**, hysteresis/state scan **17.60%**, and complexity refresh **13.41%**, with **0 prediction-checksum discrepancy**.

v1.20-R validated lazy distance materialization. At 288 anchors, cache refresh became **1.132× faster** and the complete fused step **1.025× faster**, with **0 numerical/state discrepancies** and zero hot-path distance materializations. The public distance view reconstructs exactly on access and the validated lazy cache is now the default research fused-cache implementation.

## Strongest validated engineering results

- Full native stateful execution: about **131x lower mean step time** than the frozen Python stack on the declared 1,400-frame workload.
- Peak RSS on the same workload: about **4x lower**.
- Sparse relation/causal state is allocated lazily rather than as an up-front Cartesian product.
- Regime budgets saturate exactly at configured capacity and expose pressure/unresolved states rather than silently growing.
- Native articulation is the dominant native stage in the tested scaling envelope.
- Within native articulation, cyclic transform-distance evaluation dominates measured cost.

## Strongest validated research results

### Prospective relational inference

Cortex can use learned relation structure to rank previously unseen future relations.

In the original 50-seed synthetic campaign, all scored pairs had zero direct training exposure. Heat-kernel relational potential achieved mean Hit@1 **0.995**, while direct prior, fine role distance, and common-neighbor baselines remained at tie level.

A variable-depth follow-up tested first supporting path depths 3, 5, and 7. The heat operator achieved Hit@1 **1.0** at all three depths without being given the required dependency depth in advance.

This demonstrates multi-hop relational inference in the tested synthetic worlds.

A separate structured logical-composition campaign, `kinship-composition-v1`, then trained Cortex only on solved paths of length 1–5 and tested unseen entities and unseen graph instances at lengths 6–10. Relation and answer symbols were randomly permuted per seed. Across 20 seeds, Cortex achieved **1.000 accuracy and 1.000 coverage**. Exact sequence memorization had zero test coverage and a last-relation baseline achieved **0.222 accuracy**. A hand-engineered clipped-count symbolic baseline also achieved 1.000, so the result demonstrates systematic recursive composition but does not yet establish a Cortex-specific advantage.

When one required composition transition was removed from the training examples, Cortex remained unresolved on all queries requiring it. That limitation motivated `algebra-induction-v1`.

In the follow-up campaign, 40% of finite operation tables were hidden before training. Element symbols were randomly permuted and Cortex was not told the generating algebra. It scored a preregistered candidate-law set—associativity, commutativity and a unique two-sided identity—and used only sufficiently supported laws for closure.

Across 20 seeds per family:
- cyclic C7: **0.985 coverage / 1.000 resolved accuracy**;
- dihedral D4: **1.000 coverage / 1.000 resolved accuracy**;
- non-associative subtraction mod 7: **0.000 coverage**, because no candidate law passed the evidence gate.

Associativity was discovered on every group seed and rejected on every non-associative control seed. Commutativity was correctly rejected on every D4 seed. Identity evidence was sufficient on 95% of D4 seeds; that measured miss is retained without threshold adjustment.

See [ALGEBRA_INDUCTION_V1.md](ALGEBRA_INDUCTION_V1.md).

### Adaptive predictive resolution

Predictive rate-distortion introduced the rule: keep the least complex representation that remains within an allowed predictive-distortion band.

Hysteresis reduced divergent-phase resolution switching from **32.35 to 2.65** on the frozen v1.8-R campaign while preserving predictive quality.

Local adaptive resolution reduced effective predictive complexity from **4.793 to 1.623** on the same campaign, refined only about **12.3%** of anchor-time during divergence, and returned to a fully coarse field after reconvergence.

### Exact and sparse optimization

Exact shared-state caching reproduced the previous behavior with zero measured prediction, resolution-state, and complexity discrepancy on its paired fixture while providing about **1.58x** speedup.

Sparse maintenance became clearly useful only as anchor count grew: the first tested speedup above 1.10x occurred at **72 anchors**, and the 144-anchor case reached about **1.28x** over dense sampled maintenance.

One preregistered hypothesis failed: sampled SVD at 36 anchors produced only about **1.054x**, not the expected 1.30x. The failed hypothesis is retained as a negative result.

### Fused prequential computation

Reusing one perceptual-responsibility pass throughout the prequential step produced an exact optimization of approximately:

- **2.18x** at 36 anchors;
- **2.46x** at 72;
- **2.55x** at 144;
- **2.40x** at 288.

Support truncation was behaviorally safe but much less economically useful. Even retaining only 99% responsibility mass kept roughly 64% of anchors at 288 anchors and produced only about **1.05x** additional speedup.

### Current bottleneck

v1.14-R implemented the exact optimization identified by the v1.13-R profiler. At 288 anchors, focused perceptual computation became **48.26× faster** and its profiled share fell from about **48.9%** to **2.2%**, with zero structural disagreements.

The new dominant stage is **cache refresh**, about **45.5%** of the 288-anchor fused research step. Candidate readout is second at about **21%**.

The next research engineering step is therefore **cache-refresh attribution/decomposition** before choosing another optimization.

### Hybrid semantic grounding

The isolated `hybrid-semantic-grounding-v1` Stage II-A campaign has passed its
frozen official gates. This research module keeps concept identity separate
from semantic labels and combines neural candidate-label distributions with
persistent typed relational structure.

With one adversarial swapped label pair, neural accuracy was **0.600** and
Cortex reached **0.9933 resolved accuracy at 0.9933 coverage** after 8
structural facts. With two swapped pairs, neural accuracy was **0.200** and
Cortex reached **0.960 resolved accuracy at 1.000 coverage**, with **0.950**
correction rate. False reinspection of neural-correct concepts remained zero.
Exact structural twins remained unresolved at **0.500** support rather than
being tie-broken.

This is an isolated semantic-grounding result with supplied concept IDs. The
next integration step is to feed it concepts discovered by latent concept
induction; it is not yet part of the stable native runtime.

See [HYBRID_SEMANTIC_GROUNDING_V1.md](HYBRID_SEMANTIC_GROUNDING_V1.md).

### Temporal semantic revision

The follow-up `semantic-revision-v1` campaign separates the instantaneous
grounding posterior from Cortex's accepted semantic state. Accepted labels now
have explicit hysteresis and bounded fading evidence.

Across all 60 untouched official cases:

- one-epoch contradictory neural shocks caused **0 semantic flips**;
- sustained semantic shifts revised successfully in **100%** of cases;
- every revision passed through challenged and unresolved states first;
- the original mapping recovered in **100%** of cases;
- unaffected concepts were preserved in **100%** of cases;
- maximum shift and recovery latency were both **4 epochs**.

See [SEMANTIC_REVISION_V1.md](SEMANTIC_REVISION_V1.md).

### Noisy semantic confidence and revision

The `noisy-semantic-revision-v1` campaign adds missing observations,
stochastic neural label errors, corrupted structural facts, changing evidence
reliability, intermittent contradiction, and simultaneous semantic regime
changes.

Across 60 untouched official cases:

- raw neural top-label accuracy was **0.8061**;
- Cortex top-label accuracy was **0.9639**;
- neural/Cortex Brier scores were **0.3563 / 0.0689**;
- trusted accepted-label accuracy was **0.9980** at **0.7807** coverage;
- false certainty was **0.00195**;
- **97.50%** of periods carrying a temporarily wrong old label were intercepted
  as untrusted;
- false pre-shift revision cases: **0**;
- shift and recovery success: **1.000 / 1.000**;
- p95 shift and recovery latency: **7 / 7 epochs**;
- unaffected-concept preservation: **1.000**.

A key distinction is now explicit: Cortex may retain an accepted semantic label
in memory while suspending trust in it before enough evidence exists to revise
the label itself.

See [NOISY_SEMANTIC_REVISION_V1.md](NOISY_SEMANTIC_REVISION_V1.md).

### Latent concept induction

The `latent-concept-induction-v1` campaign removes the assumption that every
useful category is supplied in advance. Cortex evaluates a bounded partition
grammar and admits shared categories only when they improve
predictive-description economy.

Across 100 untouched official cases:

- minimum two-concept partition F1: **1.000**;
- minimum three-concept partition F1: **1.000**;
- no-shared-concept control correctness: **1.000**;
- mean held-out gain over flat per-entity prediction:
  **0.0556 / 0.0288 nats per outcome**;
- minimum gain over one global concept: **0.2426 nats/outcome**;
- ambiguous-member unresolved rate: **1.000**;
- create/split/merge/retract lifecycle coverage: **1.000**;
- final concept retraction correctness: **1.000**.

The deliberately wrong geometry-decoy grouping reached at most **0.3333** F1,
while Cortex recovered the predictive grouping at **1.000** F1.

See [LATENT_CONCEPT_INDUCTION_V1.md](LATENT_CONCEPT_INDUCTION_V1.md).

### Concept-to-semantics integration

The `concept-semantic-integration-v1` campaign now composes internally
discovered latent concepts with semantic grounding.

Across 80 untouched official cases:

- minimum latent concept partition F1: **1.000**;
- raw neural semantic accuracy: **0.3333**;
- Cortex semantic accuracy / coverage: **1.000 / 1.000**;
- distractor-label resolution rate: **0.000**;
- label-causal-isolation rate: **1.000**;
- nameable-concept semantic accuracy: **1.000**;
- unnamed-concept unresolved rate: **1.000** at maximum support **0.3333**;
- split/merge semantic-lineage correctness: **1.000 / 1.000**;
- semantic invalidation after concept retraction: **1.000**;
- concept identity-lineage correctness: **1.000**.

The result establishes a clean research boundary between ontology and language:
a concept can exist without a resolved name, and a semantic belief disappears
when its concept is retracted.

See [CONCEPT_SEMANTIC_INTEGRATION_V1.md](CONCEPT_SEMANTIC_INTEGRATION_V1.md).

## Next milestones

### 1. v1.14-R — exact perceptual fast path — passed

The exact field-owned perceptual fast path passed all frozen gates.

At 288 anchors:

- focused perceptual speedup: **48.26×**;
- complete fused-step speedup: **1.96×**;
- perceptual stage share: **2.20%**;
- structural disagreements: **0**.

The new measured bottleneck is **cache refresh** at about **45.5%**. See [EXACT_PERCEPTUAL_FAST_PATH_V1_14_RESULT.md](EXACT_PERCEPTUAL_FAST_PATH_V1_14_RESULT.md).

### 2. Concept-to-semantics integration — passed

Internally discovered latent concept IDs now feed the semantic-grounding layer.
The official campaign confirmed exact causal isolation of concept identity from
semantic vocabulary, correct adversarial label correction, unresolved naming
for a real but unnamed concept, and correct semantic lineage through concept
split, merge, and retraction.

The next independent Reasoning Map axis is **explicit proof dependency**. A
natural semantic follow-up is hierarchical grounding/inheritance together with
temporal semantic revision across concept lineage.

See [CONCEPT_SEMANTIC_INTEGRATION_V1.md](CONCEPT_SEMANTIC_INTEGRATION_V1.md)
and [REASONING_MAP_V1.md](REASONING_MAP_V1.md).

### 3. v1.15-R — cache-refresh attribution / decomposition — passed

The official attribution campaign passed every frozen validity gate.

At 288 anchors:

- distance-row recomputation: **41.10%**;
- kernel regeneration: **35.50%**;
- writes/bookkeeping: **15.48%**;
- semantic features: **2.90%**;
- dirty detection: **1.88%**;
- outcome probabilities: **1.55%**;
- mean dirty-row fraction: **13.40%**.

Geometry plus kernel regeneration therefore accounts for about **76.6%** of cache-refresh cost at the largest tested scale.

The next engineering experiment should be an **exact semantic-geometry fast path**: exploit the unit-norm square-root probability features to replace dirty-row difference tensors with dot-product geometry, then derive all candidate-resolution kernel rows from the same squared-distance quantity.

See [CACHE_REFRESH_ATTRIBUTION_V1_15_RESULT.md](CACHE_REFRESH_ATTRIBUTION_V1_15_RESULT.md).

### 4. v1.16-R — exact semantic-geometry fast path — mixed official result

The exact optimization itself is validated:

- cache-refresh speedup at 288 anchors: **1.523×**;
- complete fused-step speedup: **1.174×**;
- maximum distance error: **5.55e-17**;
- maximum kernel error: **3.33e-16**;
- dirty-mask disagreements: **0**;
- structural disagreements: **0**.

However, the frozen cache-share gate required the optimized cache-refresh stage
to occupy at most **35%** of the profiled step. The official result was
**36.79%**, so the full v1.16-R milestone did not pass every frozen gate.

The implementation remains useful and exact, but the failed bottleneck-shift
hypothesis is retained.

See [EXACT_SEMANTIC_GEOMETRY_FAST_PATH_V1_16_RESULT.md](EXACT_SEMANTIC_GEOMETRY_FAST_PATH_V1_16_RESULT.md).

### 5. v1.17-R — optimized cache re-attribution — passed

The untouched official campaign reproduced the development ordering.

At 288 anchors:

- kernel regeneration: **48.63%**;
- writes/bookkeeping: **24.94%**;
- streamed semantic geometry: **13.69%**;
- semantic features: **4.31%**;
- dirty detection: **2.82%**;
- outcome probabilities: **2.32%**;
- mean dirty-row fraction: **13.28%**.

Inside streamed geometry, squared-distance accumulation is only **11.40%** and
the final square root **2.21%**. Further geometry micro-optimization is
therefore not the measured priority.

See [OPTIMIZED_CACHE_REATTRIBUTION_V1_17_RESULT.md](OPTIMIZED_CACHE_REATTRIBUTION_V1_17_RESULT.md).

### 6. v1.18-R — exact kernel-ladder fast path — rejected for performance

The candidate preserved numerical and structural behavior but did not reproduce
the development speedup on untouched official seeds.

At 288 anchors:

- kernel speedup: **1.115×** vs frozen gate **1.40×**;
- refresh speedup: **1.017×** vs frozen gate **1.10×**;
- fused-step speedup: **0.998×** vs frozen gate **1.02×**;
- maximum kernel error: **1.89e-15**;
- underflow-pattern mismatches: **0**;
- structural disagreements: **0**.

The ladder is therefore not promoted into the default cache path.

See [KERNEL_LADDER_FAST_PATH_V1_18_RESULT.md](KERNEL_LADDER_FAST_PATH_V1_18_RESULT.md).

### 7. v1.19-R — low-overhead fused-stage crossover profile — passed

The untouched official profile reproduced the development ordering with **0**
prediction-checksum discrepancy.

At 288 anchors:

- cache refresh: **37.22%**;
- candidate readout: **21.80%**;
- hysteresis/state scan: **17.60%**;
- complexity refresh: **13.41%**.

Cache refresh is therefore confirmed as the largest coarse stage rather than an
artifact of the earlier fine-grained profiler.

See [LOW_OVERHEAD_STAGE_PROFILE_V1_19_RESULT.md](LOW_OVERHEAD_STAGE_PROFILE_V1_19_RESULT.md).

### 8. v1.20-R — lazy distance materialization / write elision — passed

The hot predictive path consumes cached kernels rather than the full cached distance
matrix. The official campaign confirmed that distance synchronization can be deferred
until the distance view is explicitly requested.

The candidate must:

- keep cached features, probabilities, kernels, dirty masks, and structural
  behavior identical;
- preserve the public `distances` view by synchronizing it lazily on access;
- compare against the eager v1.16 cache step-by-step;
- measure cache-refresh and full fused-step speedup;
- include explicit tests where diagnostics request distances after arbitrary
  sequences of refreshes.

Official 288-anchor result: **1.132× refresh speedup / 1.025× fused-step speedup**, with exact distance reconstruction on demand.

See [LAZY_DISTANCE_MATERIALIZATION_V1_20_RESULT.md](LAZY_DISTANCE_MATERIALIZATION_V1_20_RESULT.md).

### 9. v1.21-R — post-lazy low-overhead stage profile

Repeat the v1.19 coarse independent-replay attribution with the promoted lazy cache. The next optimization target must be selected from this new ordering rather than stale pre-v1.20 percentages.

### 10. Freeze the next executable research specification

Once adaptive predictive geometry and relational composition stabilize, define a new frozen research specification rather than mutating `spec-v0.9.5`.

### 11. Promote validated mechanisms to Rust

Port only mechanisms accepted into that new specification, with differential contracts.

### 12. Untouched external validation

Freeze configuration before evaluation and compare against fair online/continual/graph/neural baselines on external chronological structured data.

### 13. Hybrid Cortex Stage II

Only after Stage I demonstrates value beyond the frozen encoder should Cortex feedback be allowed to influence representation learning.

## Claims discipline

Cortex currently supports claims about:

- bounded structural state;
- reversible refinement;
- prospective evidence gating;
- recurrence/adaptation/split/merge/novelty behavior;
- specific relational-inference fixtures;
- specific adaptive-resolution fixtures;
- the published native and research performance campaigns.

It does not yet support claims of universal logic, general superiority to neural networks, or broad real-world dominance.
