# Cortex project status

Last consolidated research frontier: **v1.13-R**  
Software/runtime: **v1.0.0-rc.2**  
Executable specification: **spec-v0.9.5**

This document is the canonical human-readable status page for the active repository. For the detailed system design, data flow, state boundaries, and reasoning mechanics, see [ARCHITECTURE.md](ARCHITECTURE.md). Historical experiment details remain reproducible from Git history, research code, and committed benchmark summaries.

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

The current research frontier is `v1.13-R`.

The most recent measured bottleneck is perceptual responsibility computation in the fused predictive field.

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

The v1.13-R stage-attribution campaign reports perceptual responsibility computation as the dominant fused research stage:

- 72 anchors: about **51.0%**;
- 144 anchors: about **54.0%**;
- 288 anchors: about **48.9%**.

The next research optimization should therefore be exact before approximate: avoid renormalizing stored anchors that the field constructor has already normalized.

## Next milestones

### 1. v1.14-R — exact perceptual fast path

Exploit the normalized-anchor invariant:

$$
d_i = \max\left(0, 1-a_i^\top\hat{x}\right)
$$

with only the incoming observation normalized per step.

Required gates:

- numerical prediction parity;
- zero structural disagreement;
- re-run fused-stage attribution;
- scale at least through 288 anchors.

### 2. Cortex Reasoning Map v1 — belief revision under contradiction

The cross-algebra transfer and first counterfactual-reasoning gates are now passed.

\`algebra-transfer-v1.6\` demonstrated reusable law-form transfer across independently relabeled target algebras with different orders. Mean held-out coverage gain over scratch Cortex was **+0.320 at 20% target evidence**, **+0.486 at 30%**, and **+0.537 at 40%**. The final protocol retained zero wrong resolved compatible predictions and complete abstention on random-magmas. The failed v1.5 negative-transfer campaign is preserved as part of the research history.

\`counterfactual-reasoning-v1\` then tested a different reasoning primitive. Across 140 untouched causal worlds Cortex achieved **1.000** on all frozen structural metrics: direct intervention-sensitive structure, multi-hop composition, negative reachability, edge-removal counterfactuals, redundant-path preservation, bridge cuts, missing-bridge abstention, proof-path validity, and exact state restoration.

The direct-effect evidence in this first causal benchmark is deterministic by design, so the result establishes structural reasoning rather than noisy causal discovery.

The next Reasoning Map axis is **belief revision under contradiction**:

- establish a strongly supported structural belief;
- introduce controlled counterevidence;
- require confidence to move through active → uncertain → retracted;
- reverse the regime and test recovery;
- preserve unrelated structure;
- measure retraction/recovery latency, calibration, edit count and compute.

See [REASONING_MAP_V1.md](REASONING_MAP_V1.md), [COUNTERFACTUAL_REASONING_V1.md](COUNTERFACTUAL_REASONING_V1.md), and [ALGEBRA_TRANSFER_V1_PROTOCOL.md](ALGEBRA_TRANSFER_V1_PROTOCOL.md).

These are research modules, not stable runtime semantics.

### 3. Re-profile after exact perception optimization

Do not assume the next bottleneck. Re-run attribution and follow the measured dominant stage.

### 4. Freeze the next executable research specification

Once adaptive predictive geometry and relational composition stabilize, define a new frozen research specification rather than mutating `spec-v0.9.5`.

### 5. Promote validated mechanisms to Rust

Port only mechanisms accepted into that new specification, with differential contracts.

### 6. Untouched external validation

Freeze configuration before evaluation and compare against fair online/continual/graph/neural baselines on external chronological structured data.

### 7. Hybrid Cortex Stage II

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
