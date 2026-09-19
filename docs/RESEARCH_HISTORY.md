# Cortex research history

This file keeps a compact record of the active research lineage. Detailed superseded milestone documents were removed from the active documentation surface during repository cleanup; their source, tests, result summaries, and full Git history remain available.

## v1.1-R — multiresolution semantic geometry

Introduced resolution-dependent semantic structure rather than a single fixed representational scale.

## v1.2-R — predictive relational inference

Separated semantic similarity from relation likelihood and introduced learned relational potential.

In a 50-seed synthetic prospective experiment, every evaluated pair had zero direct exposure. Mean Hit@1 was:

- direct prior: 0.250;
- fine informational distance: 0.250;
- common neighbors: 0.250;
- explicit three-hop score: 0.990;
- heat-kernel relational potential: **0.995**.

A variable-depth follow-up used first supporting path lengths 3, 5, and 7. Heat-kernel potential achieved Hit@1 1.0 at every tested depth without selecting a path order in advance.

## v1.3-R — dual geometry / tensor bridge

Explored bridging informational geometry and learned relational structure while preserving their different meanings.

## v1.4-R — learned semantic world model

Extended the research stack toward learned semantic structure rather than hand-declared world partitions.

## v1.5-R — robust articulation and latent events

Added provisional articulation, reconciliation, latent event discovery, and integrated robust latent world-model experiments.

## v1.6-R — fuzzy predictive field

Moved predictive distinctions into a continuous semantic field.

## v1.7-R — predictive rate-distortion

Selected the least complex candidate resolution that stayed within a declared predictive-distortion tolerance.

The main weakness was temporal instability: the divergent phase produced roughly 29.6 resolution changes on the original campaign.

## v1.8-R — hysteretic local adaptive resolution

Added asymmetric hysteresis and local per-anchor resolution.

Frozen 20-seed result:

- reactive global switches: 32.35;
- hysteretic global switches: **2.65**;
- reactive divergent NLL: 0.48218;
- hysteretic divergent NLL: **0.48186**;
- global divergent effective complexity: 4.793;
- local divergent effective complexity: **1.623**;
- local refined fraction: **0.123**;
- reconverged fully coarse fraction: **1.0**.

## v1.9-R — sparse cached local resolution

Separated exact cache reuse from approximate sparse refresh.

The exact cache reproduced the reference path with zero measured prediction, resolution-state, and complexity discrepancy on the paired campaign and achieved about **1.576x** speedup.

The sparse path reached about **1.674x** over the old reference while refreshing about 12.2% of kernel rows, scanning about 32.1% of anchors, and invoking SVD on about 3.27% of observations.

## v1.10-R — resolution scaling crossover

Swept 9, 18, 36, 72, and 144 anchors.

Sparse maintenance became beneficial only at larger fields:

- 9: 0.950x;
- 18: 0.999x;
- 36: 1.049x;
- 72: **1.140x**;
- 144: **1.281x**.

The preregistered 1.30x sampled-SVD hypothesis at 36 anchors failed; measured speedup was about **1.054x**.

## v1.11-R — fused support-sparse predictive application

The exact optimization was prequential fusion: compute perceptual responsibilities once and reuse them across candidate prediction, local prediction, regional-loss attribution, and field update.

Speedup over the prior path:

- 36 anchors: 2.184x;
- 72: 2.459x;
- 144: 2.550x;
- 288: 2.403x.

At retained mass q=0.999, support compression preserved behavior extremely closely but retained about 87% of anchors and added essentially no speed.

## v1.12-R — retained-mass support tradeoff

Swept q in {0.99, 0.995, 0.999}.

At 288 anchors, q=0.99 retained about 64.1% of anchors and remained behaviorally admissible with very small prediction error, but improved runtime by only about **1.05x** over fused full support.

Conclusion: support truncation was limited by implementation economics rather than predictive fidelity.

## v1.13-R — fused-stage attribution

Measured the fused prequential step internally at 72, 144, and 288 anchors.

Perceptual responsibility computation is the dominant stage at every scale, around **49–54%** of measured runtime. Cache refresh is second, around 20–24%.

The next research target is an exact perceptual fast path using the invariant that stored anchors are already normalized.

## Current direction

Two immediate scientific directions are now well motivated:

1. **v1.14-R exact perceptual optimization**, followed by re-profiling;
2. **Cortex Reasoning Map v1**, with cross-algebra transfer and structural counterfactual reasoning now passed, and belief revision under contradiction next.

The Reasoning Map then proceeds toward structural analogy/isomorphism, latent-concept induction, explicit proof dependency, planning/replanning, and open-world novelty attribution. See [REASONING_MAP_V1.md](REASONING_MAP_V1.md).


## kinship-composition-v1 — structured logical composition

This parallel reasoning campaign tested whether Cortex can induce and recursively apply a typed relation algebra beyond the depth of its solved training examples.

Protocol:

- solved training paths: lengths 1–5;
- test paths: lengths 6–10;
- all test entities and graph instances unseen;
- distractor branches and disconnected facts present;
- primitive relation symbols permuted per seed;
- answer labels permuted per seed;
- 20 deterministic seeds.

Mean results:

- Cortex relation algebra: **1.000 accuracy / 1.000 coverage**;
- exact sequence memory: **0.000 accuracy / 0.000 coverage**;
- last-relation baseline: **0.222 accuracy / 1.000 coverage**;
- hand-engineered clipped-count baseline: **1.000 accuracy / 1.000 coverage**.

A missing-rule diagnostic omitted the transition required to derive the aunt/uncle state. Cortex then remained unresolved on every query requiring that transition instead of guessing.

Interpretation: Cortex can perform recursive compositional generalization once the local relation laws have been evidenced. The test does not show derivation of a completely unseen law, and the hand-engineered symbolic baseline demonstrates that this finite algebra is solvable with an appropriate inductive bias. Neural/GNN comparison remains the next scientific discriminator.


## algebra-induction-v1 — evidence-gated algebraic law induction

This campaign tested whether Cortex could derive missing binary-operation results from structural laws that were themselves selected from partial evidence.

The system received anonymous, randomly permuted operation symbols with **40% of the table hidden**. It was not told the generating algebra. The preregistered candidate-law vocabulary contained associativity, commutativity and existence of a unique two-sided identity.

Twenty seeds were run for each of three families:

- cyclic C7;
- noncommutative dihedral D4;
- subtraction mod 7 as a non-associative control.

Mean held-out results:

- cyclic C7: **0.985 accuracy / 0.985 coverage / 1.000 resolved accuracy**;
- dihedral D4: **1.000 / 1.000 / 1.000**;
- subtraction mod 7: **0.000 / 0.000 / 0.000**.

Direct table memory had zero held-out coverage in every family. A majority-product baseline achieved only about 0.06 accuracy.

Law selection was also structurally correct:

- C7: associativity 1.00, commutativity 1.00, identity 1.00;
- D4: associativity 1.00, commutativity 0.00, identity 0.95;
- subtraction control: all three 0.00.

The single D4 identity miss is retained as measured; no threshold was changed after observing it. Associativity alone was sufficient to recover every D4 held-out product.

Interpretation: Cortex can now select among candidate algebraic laws from evidence and derive previously unseen products through exact closure. The law *forms* are still supplied in advance, so this is not yet open-ended theorem or axiom discovery.

See [ALGEBRA_INDUCTION_V1.md](ALGEBRA_INDUCTION_V1.md).


## algebra-form-induction-v1.7 — bounded fuzzy algebraic form discovery

This campaign removed the named candidate-law vocabulary used by `algebra-induction-v1`.

Cortex received an anonymous partial operation table plus a bounded grammar over `x`, `y`, `z` and one binary operator. Canonicalization under variable renaming and equation-side exchange produced **375 candidate equation forms**.

The final fuzzy model separates:

- structural law membership;
- predictive activation;
- empirical applicability scope.

The distinction emerged from failed intermediate protocols v1 through v1.6, all retained in the source/report rather than hidden by threshold changes.

Official hosted campaign:

- 20 seeds per family;
- 6 families;
- 120 cases;
- 40% table holdout.

Mean Cortex held-out coverage / resolved accuracy:

- cyclic C7: **0.945 / 1.000**;
- dihedral D4: **0.867 / 1.000**;
- min semilattice: **0.589 / 1.000**;
- left-zero semigroup: **0.979 / 1.000**;
- subtraction mod 7: **0.700 / 1.000**;
- random magmas: **0.000 / unresolved**.

Target-form behavior:

- rebracketing discovered in 100% of cyclic and dihedral seeds;
- operand swap discovered in 100% of cyclic seeds and rejected in 100% of dihedral seeds;
- repetition discovered in 100% of semilattice seeds;
- left projection discovered in 100% of left-zero seeds;
- rebracketing rejected in 100% of subtraction and random-magma seeds.

Across all 120 cases there were **zero wrong resolved Cortex predictions** and **zero closure conflicts**. Every random magma remained completely unresolved.

Interpretation: Cortex can discover useful equation forms from a bounded grammar, distinguish law support from applicability, and reuse supported forms for conservative exact closure. This remains bounded form discovery, not unrestricted theorem proving.

See [ALGEBRA_FORM_INDUCTION_V1.md](ALGEBRA_FORM_INDUCTION_V1.md).


## algebra-transfer-v1.6 — cross-world abstract law transfer

This campaign tested whether equation forms discovered from several anonymous
source algebras can become reusable abstractions in independently relabeled
target algebras with different orders.

The transfer object contains canonical equation forms and source evidence
metadata, but no source element identities, table cells, or applicability
basins. Every transferred law is re-grounded against target evidence.

The development history is scientifically important. The first official v1.5
campaign showed strong compatible transfer but failed the random-magma safety
controls because adaptively searching many candidate bundles made short
zero-error streaks possible by chance. That failed result is retained.

v1.6 added a multiplicity-aware evidence requirement and moved official
evaluation to untouched target/control seeds 100–109.

The v1.6 campaign passed all frozen gates across 170 rows.

Mean compatible held-out coverage:

- 20% target evidence: transfer **0.4363** vs scratch **0.1167**;
- 30%: **0.6935** vs **0.2074**;
- 40%: **0.8397** vs **0.3026**.

Coverage gain over scratch was therefore:

- **+0.3196** at 20%;
- **+0.4862** at 30%;
- **+0.5371** at 40%.

Safety behavior:

- zero wrong resolved compatible predictions;
- zero closure conflicts;
- zero resolved predictions on random-magmas;
- subtraction rejected the imported rebracketing law;
- source and target labels/orders were disjoint.

Interpretation: within the declared finite fixtures, a discovered equation form
can become a reusable cross-world abstraction that reduces target evidence
requirements while remaining falsifiable on incompatible worlds.


## counterfactual-reasoning-v1 — structural causal counterfactual reasoning

This is the first completed experiment in **Cortex Reasoning Map v1**.

Cortex learned directed influence from deterministic controlled local
intervention/control probes, then reasoned over the learned epistemic graph.

Seven world families and untouched seeds 100–119 produced **140 official
worlds**.

All frozen gates passed:

- direct causal resolved precision / coverage: **1.000 / 1.000**;
- causal-edge recall: **1.000**;
- null-edge recall: **1.000**;
- correlation-trap rejection: **1.000**;
- multi-hop accuracy / coverage: **1.000 / 1.000**;
- direct-edge baseline multi-hop accuracy: **0.000**;
- negative-query accuracy: **1.000**;
- edge-removal counterfactual accuracy / coverage: **1.000 / 1.000**;
- redundant-path preservation: **1.000**;
- bridge-cut success: **1.000**;
- missing-bridge unresolved rate: **1.000**;
- proof-path validity: **1.000**;
- state-restoration rate: **1.000**.

The evidence was intentionally deterministic, so this benchmark establishes the
structural reasoning operation rather than noisy real-world causal discovery.

Interpretation: Cortex can temporarily alter a learned causal world, reason
inside the altered structure, preserve epistemic uncertainty when a required
bridge is unresolved, and return to the exact original state afterward.

See [COUNTERFACTUAL_REASONING_V1.md](COUNTERFACTUAL_REASONING_V1.md).
