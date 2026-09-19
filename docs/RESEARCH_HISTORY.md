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
2. **systematic relational generalization**, extending the passed kinship depth-extrapolation gate toward held-out rule induction, conflicting evidence, multi-path proofs, and neural/GNN comparison.


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
