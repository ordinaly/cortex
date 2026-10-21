# Cortex project status

Last consolidated research frontier: **v1.22-R**  
Software/runtime: **v1.0.0-rc.2**  
Executable specification: **spec-v0.9.5**

This document is the canonical human-readable status page for the active repository. For the detailed system design, data flow, state boundaries, and reasoning mechanics, see [ARCHITECTURE.md](ARCHITECTURE.md). Historical experiment details remain reproducible from Git history, research code, and committed benchmark summaries. The prioritized cross-track roadmap is maintained in [PROJECT_MILESTONES.md](PROJECT_MILESTONES.md).

## Active architecture direction — Autonomous Core + LPU v0.1

The main Cortex path is now being simplified around **Cortex-owned persistent
memory and autonomous structural reasoning**. A neural language model is no
longer part of the active cognitive loop.

The first implementation of this pivot adds:

- direct chronological structured-dataset exposure through
  `ObservationFrame` and `train_dataset(...)`;
- a small stateless **Logical Processing Unit (LPU)** for exact local Boolean,
  comparison, rational-arithmetic, and reachability operations;
- an explicit boundary in which the LPU cannot own facts, predicates, ontology,
  confidence, or persistent memory.

The former Ollama/Qwen language gateway, active hybrid runtime, and hybrid
benchmark workflow have been removed from the active package path. Their Git
history and selected research reports/results are retained as experimental
provenance.

The immediate scientific question is now **teacher-free autonomy**: after a
chronological learning stream ends, what can Cortex itself retain, compose,
revise, transfer, and keep calibrated without a neural model participating in
reasoning?

See [AUTONOMOUS_CORE_LPU_V0_1.md](AUTONOMOUS_CORE_LPU_V0_1.md).

## General Knowledge Feed v0.1

A neutral structured-knowledge boundary is now under validation in PR #80.
External properties are represented as ordinary data entities inside a reified
statement grammar rather than as fixed Cortex predicates.

The first 22-statement smoke campaign passed all storage/retrieval gates:
**129/129 unique symbols bound injectively, 0 binding conflicts, 12/12 exact
retrieval queries, preserved multi-value relations, and identical query answers
after statement-order permutation**.

The Wikidata adapter preserves multiple values, qualifiers, ranks, reference
hashes, and typed literal payloads in the external Knowledge Packet. Rank,
references, and provenance are not yet mapped into Cortex confidence.

See [GENERAL_KNOWLEDGE_FEED_V0_1.md](GENERAL_KNOWLEDGE_FEED_V0_1.md).

## Relational Hypergeometry v0.1

PR #80 now includes a read-only soft hypergraph geometry over Cortex concept
basins. Subject, relation, and object remain ordinary fuzzy concepts; one
statement becomes a weighted higher-order edge joining all three.

The geometry provides relation-conditioned local directions, maximum-bottleneck
connectivity, a resolution distance $d_R = 1 - \kappa$, and reversible
H0-style components controlled by evidence resolution. Multi-valued relations
remain explicitly supported.

This mechanism is **research-only** and does not modify the frozen runtime
semantics or native entity binding.

See [RELATIONAL_HYPERGEOMETRY_V0_1.md](RELATIONAL_HYPERGEOMETRY_V0_1.md).

## Hypergeometry + LPU Stress v0.2

A follow-up research campaign now validates the LPU as an exact execution layer
for Cortex-articulated relational compositions.

Across 40 clean depth-2 through depth-5 cases, Cortex-only and Cortex+LPU agreed
on every answer. LPU witnesses were valid in every case and persistent Cortex
memory was unchanged. At depth 5 / width 8, path enumeration represented 4096
complete paths while the layered LPU graph required 211 edges; hosted-runner
execution characterization was approximately 24.8 ms versus 0.676 ms.

The LPU remained subordinate to Cortex confidence: 20% contradictory evidence
was tolerated at learned confidence 0.8, while 40% contradiction caused Cortex
to withhold the rule and no LPU call was issued.

See [HYPERGEOMETRY_LPU_STRESS_V0_2.md](HYPERGEOMETRY_LPU_STRESS_V0_2.md).

## Articulation Controller v0.1

The research stack now includes an epistemic controller between Cortex's fuzzy
relational hypergeometry and exact execution.

Its top-level states are `EXPLORE`, `EXECUTE`, and `UNRESOLVED`. An
`EXECUTE` decision independently chooses native Cortex traversal or the LPU
from structural path-compression estimates and exact-witness requirements.

Native-runtime validation across 10 randomized seeds / 40 clean depth-2 through
depth-5 cases achieved 100% correct execution and 100% backend-policy
agreement. Depths 2-3 remained native; depths 4-5 routed to the LPU. Moderate
20% contradiction remained executable at confidence 0.8, while 40%
contradiction was unresolved on every seed. Ambiguous source articulation
remained in exploration.

See [ARTICULATION_CONTROLLER_V0_1.md](ARTICULATION_CONTROLLER_V0_1.md).

## Articulation Controller Calibration v0.2

The fixed v0.1 articulation policy now has a learned, auditable follow-up.
v0.2 fits small monotone decision surfaces from held-out consequences rather
than copying hand-selected controller thresholds.

The calibrated policy separates four questions:

- is current evidence ready for exact execution?
- is rejected execution caused by persistent relational conflict?
- is it caused by persistent structural weakness?
- if execution is admitted, is native traversal or the LPU the cheaper exact
  backend?

On 1,200 held-out future-evidence episodes, execution precision was **0.9405**
with a **0.0125 dangerous false-execute rate**. The relational-conflict surface
reached **0.9747 recall** and the structural-weakness surface **0.9912 recall**.
Overall three-state accuracy was **0.7075**, with errors intentionally biased
toward continued exploration rather than premature execution.

A separate native Cortex holdout produced **40/40 correct clean executions**,
100% memory isolation, zero binding conflicts, and learned knowledge resolution
**epsilon = 0.65** in every clean case. Moderate 20% contradiction remained
open as `EXPLORE`, while 40% contradiction became `UNRESOLVED`. The
measured backend calibration selected Cortex for 10 clean cases and LPU for 30
in the successful CI run.

See [ARTICULATION_CONTROLLER_CALIBRATION_V0_2.md](ARTICULATION_CONTROLLER_CALIBRATION_V0_2.md).

## Cortex Autonomy Campaign v1

The first teacher-removal campaign passed every preregistered gate across five
randomized native-runtime worlds.

The teacher was detached after 76 chronological statements per world. After
detachment there were **0 model calls, 0 held-out answer leaks, and 0 binding
conflicts**. Calibrated Cortex achieved **1.000 coverage / 1.000 accuracy** on
both childhood-held-out queries and entirely new post-detachment entities.
Legitimate multi-value changes were preserved.

Under changed-world evidence, the learned controller followed
`EXECUTE -> EXPLORE -> UNRESOLVED -> EXECUTE` on every seed while the
no-controller ablation never abstained. A disjoint relation family remained
correct throughout, demonstrating local revision on this fixture.

A depth-four / width-four autonomous query was correct on every seed. The
learned controller selected the LPU in every case, exact witnesses were valid,
and hosted-runner execution was **1.717x faster at the median** than forcing the
same articulation through the Cortex enumerating executor.

The memory-pressure sweep exposed one negative result: a 256-entity runtime hit
a `ValueError`, while 512 and 1024 entity capacities retained the query with
408 active entities. Graceful pressure handling at the structured-knowledge
boundary remains open.

See [CORTEX_AUTONOMY_CAMPAIGN_V1.md](CORTEX_AUTONOMY_CAMPAIGN_V1.md).

## Plasticity Frontier v1

The first dedicated plasticity characterization quantifies a large
re-articulation dead zone in the frozen cumulative-evidence + calibrated-v0.2
stack.

Across seven randomized worlds, replacing a learned composition required about
**17.8x as many NEW observations as accumulated OLD observations** before the
replacement relation could reach `EXECUTE`. Prior masses
4 / 8 / 16 / 32 / 64 required approximately
72 / 143 / 285 / 570 / 1139 new examples.

The controller detected change much earlier: the old relation left
`EXECUTE` after only 1 / 1 / 1 / 2 / 4 contradictory examples and became
`UNRESOLVED` after 3 / 5 / 9 / 18 / 36. Thus the dominant weakness is not
novelty detection but slow **re-articulation after a genuine regime shift**.

A disjoint control relation remained fully stable. Short contradictory bursts
also exposed asymmetric recovery cost: at 32 prior examples, bursts of
1 / 2 / 4 / 8 required 0 / 4 / 40 / 111 later supporting observations to
return to `EXECUTE`.

The follow-up **Relational Regime Plasticity v0.1** Pareto campaign swept
270 policies; 105 were feasible and 2 were Pareto-optimal. The calibration and
CI gates passed, and the selected knee is frozen as the research default. The selected knee
uses fast decay 0.50, pressure decay 0.90, suspicion ratio 0.05, and a 5-step
confirmation streak. It preserves the exact v0.2 change-detection lags while
switching to the replacement regime in **9 observations for every tested prior
mass**, with zero declared false switches and exact historical retention.

See [PLASTICITY_FRONTIER_V1.md](PLASTICITY_FRONTIER_V1.md) and
[RELATIONAL_REGIME_PLASTICITY_V0_1.md](RELATIONAL_REGIME_PLASTICITY_V0_1.md).

## Autonomy Robustness & Temporal Plasticity v1

**Phase B passed; the declared research program is complete.**

Phase A preserved the frozen Relational Regime Plasticity v0.1 behavior and
measured two weaknesses: stale exact execution under missingness and complete
source-reliability blindness in a trusted-minority / weak-majority fixture.

Phase B keeps durable historical evidence exact while adding:

- bounded odds-relative weighting for the fast/current evidence channel;
- reliability-scaled effective evidence time and contradiction pressure;
- a read-only execution-freshness boundary with one missing-event grace.

Across 12 seeds, the selected source policy resolved the original source trap
to NEW in **12/12** runs with mean switch step **15**; unweighted and linear
controls remained on OLD in **12/12**. A separate source holdout also resolved
NEW in **12/12**, while frequent weak contradictory noise caused **0 false
switches**.

The selected freshness boundary preserves the clean **9 observed-event** switch.
At 30% missingness, unsafe exact execution fell from **0.1626 to 0.1458**; at
60% missingness it fell from **0.1860 to 0.1082**. Stable-world safe-execution
coverage remained **0.9114 / 0.6422** at 30% / 60% missingness.

All full-trust non-regression gates passed: corrupted-shift p95 remained
**26 / 58** at 10% / 20% corruption, stationary noise produced zero false
switches, compound locality held, raw source history remained exact, and 48
fixed-vocabulary contexts remained at 192 regime cells.

See
[AUTONOMY_ROBUSTNESS_TEMPORAL_PLASTICITY_V1.md](AUTONOMY_ROBUSTNESS_TEMPORAL_PLASTICITY_V1.md).


## Open-World Robustness v1 — closed-loop synthesis v0.1

Open-World Robustness v1 remains the active autonomy investigation. OWR-A
through OWR-D were first characterized on the frozen Cortex, exposing evidence
dependence, temporal provenance, compound explanation, and bounded-memory
failures. OWR-E remains untouched.

The first shared architectural response is now implemented as the research-only
**Closed Articulation Loop v0.1**. It completes the missing interaction:

```text
foreground articulation
    -> downward consolidation
    -> active / latent bounded memory
    -> conditioned activation
    -> guarded bubbling
    -> future articulation
```

The mechanism has no OWR stage labels, no hidden source truth, and no self-model.
It infers provenance redundancy from repeated public agreement, separately
learns whether reports form temporally separated evidence episodes, preserves
late evidence as history without automatically rewriting foreground state,
forms generic compound residual candidates, reactivates sufficiently supported
historical basins only after fresh confirmation, and coarsens low-utility memory
distinctions under pressure.

On one unchanged policy across 12 development worlds per stage:

- **OWR-A:** copied-majority traps improved from frozen safe coverage 0/48 to
  **33/48 correct, 0 wrong** (0.6875 coverage, 1.000 resolved accuracy).
- **OWR-B:** overall selective risk fell from about **0.319 to 0.102**. The
  three near-transition risks fell from **1.000 / 1.000 / 0.979** to
  **0.619 / 0.280 / 0.179**. This is a large improvement but the first
  transition remains unsafe and OWR-B is not solved.
- **OWR-C:** local relation coverage/accuracy remains **1.000 / 1.000** and
  every world forms generic multi-subject compound background candidates.
  Semantic component labels are intentionally not emitted yet.
- **OWR-D:** the research memory completes every 3,000-event stream with
  **256 stored basins**, about **2254 coarsened distinctions** on average,
  **0.7756 query coverage**, **1.000 resolved accuracy**, and no hard
  research-layer failure.

The OWR-D result does **not** remove the native entity-capacity cliff: native
structured ingestion still allocates physical entities before the research
memory can coarsen them. Native integration is a separate engineering step.

No official OWR seeds have been generated, performance gates remain unfrozen,
and OWR-E has not been used for tuning.

See [CLOSED_ARTICULATION_LOOP_V0_1.md](CLOSED_ARTICULATION_LOOP_V0_1.md),
[OPEN_WORLD_ROBUSTNESS_A_DEVELOPMENT.md](OPEN_WORLD_ROBUSTNESS_A_DEVELOPMENT.md),
and
[OPEN_WORLD_ROBUSTNESS_BCD_DEVELOPMENT.md](OPEN_WORLD_ROBUSTNESS_BCD_DEVELOPMENT.md).

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

The research line has moved beyond the frozen runtime semantics along multiple independent tracks. Predictive semantic geometry and adaptive representational resolution remain at `v1.22-R`, while autonomous relational learning now uses **Relational Regime Plasticity v0.1** as its frozen research default.

The **Autonomy Robustness & Temporal Plasticity v1** cross-track milestone has completed Phase B and passed its declared research gates.

v1.16-R implements an exact streamed semantic-geometry cache path selected by the v1.15-R attribution campaign. On untouched official seeds at 288 anchors, cache refresh became **1.52× faster** and the complete fused step **1.17× faster**, with maximum cached-distance error **5.55e-17**, maximum kernel error **3.33e-16**, and **0 structural disagreements**. The frozen milestone nevertheless did not pass every gate: cache refresh remained **36.79%** of the profiled step, above the preregistered **35%** bottleneck-shift ceiling.

v1.17-R then re-attributed the optimized cache. At 288 anchors, **kernel regeneration was 48.63%** of instrumented refresh, writes/bookkeeping **24.94%**, and streamed geometry **13.69%**.

v1.18-R tested an exact scale-gated kernel ladder. Numerical behavior remained effectively identical (maximum kernel error **1.89e-15**, zero underflow/dirty/structural mismatches), but the untouched official performance campaign rejected promotion: at 288 anchors the kernel path was only **1.115×** faster, refresh **1.017×**, and the complete fused step **0.998×**, below all frozen performance gates. The candidate remains research-only.

v1.19-R then re-profiled the fused research path with one timed stage per fresh replay to reduce observer interference. The untouched official profile reproduced development: at 288 anchors, cache refresh was **37.22%**, candidate readout **21.80%**, hysteresis/state scan **17.60%**, and complexity refresh **13.41%**, with **0 prediction-checksum discrepancy**.

v1.20-R validated lazy distance materialization. At 288 anchors, cache refresh became **1.132× faster** and the complete fused step **1.025× faster**, with **0 numerical/state discrepancies** and zero hot-path distance materializations. The public distance view reconstructs exactly on access and the validated lazy cache is now the default research fused-cache implementation.

v1.21-R repeated the low-overhead independent-replay profiler after that promotion. On untouched official seeds at 288 anchors, cache refresh remains the largest coarse stage at **35.55%**, followed by candidate readout at **21.21%**, hysteresis/state scan at **19.09%**, and complexity refresh at **13.94%**. Prediction-checksum discrepancy remained **0**.

v1.22-R then tested single-buffer kernel refresh / allocation elision. The candidate preserved exact predictions, state, kernels, dirty masks, and lazy distance behavior across all development cases, but it was slower rather than faster. At 288 anchors, median refresh speedup was **0.990×** and median complete fused-step speedup **0.988×**, with zero fused-step wins across six development seeds. The candidate was not promoted. Because development failed the frozen performance gates, official seeds 1200–1204 were not run.

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

### Current performance frontier

v1.21-R remains the authoritative coarse profile of the promoted research path:
at 288 anchors, cache refresh is **35.55%**, candidate readout **21.21%**,
hysteresis/state scan **19.09%**, and complexity refresh **13.94%**.

v1.22-R tested whether the remaining cache cost could be reduced by replacing
the temporary resolution-by-dirty-row-by-anchor kernel block with a reusable
two-dimensional buffer. The implementation was exact but development rejected
it economically: refresh was **0.990×** and the full fused step **0.988×** the
reference speed at 288 anchors.

The validated v1.20 lazy cache therefore remains the default. Another
layout-only cache rewrite is not currently justified. If optimization work
continues, candidate readout is the next-largest measured coarse stage.

See [SINGLE_BUFFER_KERNEL_REFRESH_V1_22_RESULT.md](SINGLE_BUFFER_KERNEL_REFRESH_V1_22_RESULT.md).

### Historical experiment — hybrid semantic grounding

This result is retained as experimental provenance and is **not part of the active autonomous architecture**. The isolated `hybrid-semantic-grounding-v1` Stage II-A campaign passed its
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

### Explicit proof dependency

The `explicit-proof-dependency-v1` campaign now upgrades reasoning provenance
from one valid path to an explicit finite proof/dependency DAG.

Across **120 untouched official cases**, every frozen gate passed:

- proof-DAG and provenance validity: **1.000 / 1.000**;
- shortest-proof selection: **1.000**;
- redundant alternate-proof survival: **1.000**;
- deeper-proof takeover: **1.000**;
- shared-root invalidation: **1.000**;
- contradictory-proof conflict handling: **1.000**;
- nested partial invalidation: **1.000**;
- irrelevant-premise locality: **1.000**;
- exact restoration: **1.000**.

The single-proof baseline failed all designated alternate-support cases.

This remains a finite supplied-rule benchmark, not unrestricted theorem
discovery.

See [PROOF_DEPENDENCY_V1.md](PROOF_DEPENDENCY_V1.md).

### Planning / replanning

The `planning-replanning-v1` campaign now connects explicit dependency
tracking to action selection and repair.

Across **140 untouched official cases**, every frozen gate passed:

- repair-mode and impact-localization accuracy: **1.000 / 1.000**;
- feasible optimal-cost agreement: **1.000**;
- local repair / immediate rejoin: **1.000 / 1.000**;
- prefix / suffix preservation: **1.000 / 1.000**;
- cost adaptation and reversible restoration: **1.000 / 1.000**;
- irrelevant-change stability: **1.000**;
- global-fallback correctness: **1.000**;
- infeasible detection: **1.000**.

On 80 local-change cases, local repair expanded **2.425 states** on average
versus **27.5** for full replanning, a mean ratio of **0.08745**, while matching
the full-search optimum.

See [PLANNING_REPLANNING_V1.md](PLANNING_REPLANNING_V1.md).

### Open-world novelty attribution

The `open-world-novelty-attribution-v1` campaign now distinguishes the kind of
structural explanation required by surprising evidence.

Across **160 untouched official cases**:

- exact attribution accuracy: **1.000**;
- resolved/unresolved correctness: **1.000**;
- exact edit locality: **1.000**;
- false persistent edit rate: **0.000**;
- post-edit stability: **1.000**;
- ambiguous-case unresolved rate: **1.000**;
- transient recovery/no-edit correctness: **1.000**.

The frozen baselines reached **0.375** and **0.125** exact attribution accuracy.

The result covers noise, new instances, new entity types, new relations, and
new regimes under finite supplied evidence. It does not yet cover compound or
adversarial novelty.

See [OPEN_WORLD_NOVELTY_ATTRIBUTION_V1.md](OPEN_WORLD_NOVELTY_ATTRIBUTION_V1.md).

### Novelty-aware replanning

The `novelty-aware-replanning-v1` campaign now composes open-world novelty
attribution with planning dependency localization.

Across **160 untouched official cases**:

- attribution accuracy: **1.000**;
- novelty-to-plan decision accuracy: **1.000**;
- model-component locality: **1.000**;
- repair optimality: **1.000**;
- exact plan preservation and zero-search rate on irrelevant novelty:
  **1.000 / 1.000**;
- local impact localization: **1.000**;
- local prefix/suffix preservation: **1.000 / 1.000**;
- mean local/full expansion ratio: **0.03846**;
- regime-global correctness: **1.000**;
- unresolved deferral safety: **1.000**.

The integration now supports four explicit responses to surprise: preserve,
local repair, global replan, or defer.

See [NOVELTY_AWARE_REPLANNING_V1.md](NOVELTY_AWARE_REPLANNING_V1.md).

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

Novelty-aware replanning has now passed. The next scientific milestone is
**Open-World Robustness v1**. A natural semantic follow-up remains
hierarchical grounding/inheritance together with temporal semantic revision
across concept lineage.

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

### 9. v1.21-R — post-lazy low-overhead stage profile — passed

The untouched official profile retained cache refresh as the largest coarse stage.

At 288 anchors:

- cache refresh: **35.55%**;
- candidate readout: **21.21%**;
- hysteresis/state scan: **19.09%**;
- complexity refresh: **13.94%**;
- prediction-checksum discrepancy: **0**.

See [POST_LAZY_STAGE_PROFILE_V1_21_RESULT.md](POST_LAZY_STAGE_PROFILE_V1_21_RESULT.md).

### 10. v1.22-R — single-buffer kernel refresh / allocation elision

The current lazy cache still materializes a temporary tensor of all dirty kernel rows across every resolution before copying those rows into the persistent kernel cache.

Test an exact candidate that computes one resolution at a time into a reusable two-dimensional scratch buffer, then commits the symmetric row/column updates before reusing that buffer for the next resolution.

The experiment must keep the existing lazy cache as the oracle and require:

- kernel/prediction/state parity;
- zero structural disagreements;
- no change to lazy distance semantics;
- focused refresh timing;
- full fused-step paired timing;
- explicit allocation/temporary-size accounting.

Promotion requires a clear paired fused-step gain. A fine-grained internal timing improvement alone is insufficient.

### 11. Freeze the next executable research specification

Once adaptive predictive geometry and relational composition stabilize, define a new frozen research specification rather than mutating `spec-v0.9.5`.

### 12. Promote validated mechanisms to Rust

Port only mechanisms accepted into that new specification, with differential contracts.

### 13. Untouched external validation

Freeze configuration before evaluation and compare against fair online/continual/graph/neural baselines on external chronological structured data.

### 14. Hybrid Cortex Stage II

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

## Possibility Geometry v0.1 — finite research oracle

An opt-in [possibility reference layer](CORTEX_POSSIBILITY_GEOMETRY_V0_1.md) now implements exact rational
probability transport, observation-conditioned updates, coherent finite-trace
probes, predictive neighbourhoods, and an exact partition closure check.
It preserves competing explanations and reports impossible observations explicitly.
Candidates and kernels are supplied; autonomous discovery, native runtime wiring,
and compression remain deferred. Existing geometric compatibility scores are not
automatically interpreted as environment transition probabilities.

## Question learning v0.2 — bounded research experiment

The [question-learning experiment](CORTEX_QUESTION_LEARNING_V0_2.md) learns conditional outcome laws from
acquired observations, validates one proposed distinction on fresh evidence, and
uses the finite possibility reference to choose paid questions. Ten held-out
learnable seeds show 10.12% decision error versus 25.10% for random questioning
at the same evaluation query count; ten noise-only controls reject refinement.
Initial learning still queries both supplied probes. Sensor invention, noisy
observation models, online structural revision and native integration are deferred.
