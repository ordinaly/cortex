# Cortex characterization-v1

`characterization-v1` is a ground-truthed synthetic stress campaign for Cortex. Its purpose is to map behavioral strengths, weaknesses, and phase boundaries without tuning Cortex to the fixtures.

The campaign is deliberately separate from the performance benchmarks. A passing case is finite evidence about the tested range, not a proof of general continual-learning performance.

## Provenance

- Cortex release: `1.0.0-rc.2`
- frozen executable specification: `spec-v0.9.5`
- benchmark protocol: `characterization-v1`
- campaign source/head: `a8fab3b4295713b329190bd5289576786bd0a42b`
- Python runtime reported by the campaign: `3.13.15`
- workflow run: `35194877409`
- raw artifact: `cortex-characterization-v1`
- raw artifact digest: `sha256:e9fb3986ad62d34bacc2f76e9c2b4c935214a21260d26c4128115de70e44c532`
- 50 cases x 3 deterministic repetitions = 150 result rows

Each result row records the software/spec/protocol versions, exact Git SHA, seed, resolved configuration hash, and case parameters.

## What is tested

The campaign has six suites.

1. **Recurrence envelope** — four recurring latent regimes are swept over RMS separation and observation noise. Cortex is compared with a deliberately simple bounded online nearest-centroid baseline using Cortex's configured recurrence tolerance.
2. **Novelty detection** — a third regime is introduced after two familiar regimes have stabilized. The campaign measures detection rate, delay, and false discoveries.
3. **Gradual drift** — one of three recurring regimes moves continuously while the others remain fixed. The campaign measures identity purity and structural fragmentation.
4. **Hard budget overload** — more mutually separated regimes are presented than the configured regime budget can store.
5. **Full-stack entity binding** — multiple entities are simultaneously observed under quarter-cycle nuisance transformations, additive noise, and sparse feature corruption. Ground-truth entity IDs are used only for scoring.
6. **Long-run continual stability** — an 8,000-step recurring stream with reversible drift checks finite state, numerical health, budget behavior, and latency.

The baseline is a sanity comparator, not a state-of-the-art continual-learning baseline. The full-stack identity suite currently has no external baseline.

## Reading the identity metrics

Cluster labels are arbitrary, so the campaign does not compare numeric IDs directly. `purity`, `fragmentation`, and `collision` must be read together:

- purity near 1 means each learned identity is mostly associated with one true identity;
- fragmentation near 1 means each true identity is represented by one learned identity;
- collision near 1 means each learned identity represents one true identity.

High purity with high fragmentation therefore indicates over-segmentation, not successful identity tracking.

## Main results

### 1. Recurrence has a clear separation boundary

The configured recurrence tolerance in these runs is `0.38` RMS. Median results across three repetitions were:

| latent RMS separation | noise range | Cortex purity | Cortex fragmentation | interpretation |
|---:|---:|---:|---:|---|
| 0.20 | 0.02-0.18 | 0.25 | 1.00 | all four regimes collapse into one representation |
| 0.35 | 0.02 | 0.35 | 1.75 | ambiguous boundary; partial separation with fragmentation |
| 0.35 | 0.08 | 0.467 | 1.75 | ambiguous boundary |
| 0.35 | 0.18 | 0.742 | 1.75 | more separation, but still fragmented |
| 0.50 | 0.02-0.18 | 1.00 | 1.00 | perfect regime separation after warm-up |
| 0.75 | 0.02-0.18 | 1.00 | 1.00 | perfect regime separation after warm-up |
| 1.00 | 0.02-0.18 | 1.00 | 1.00 | perfect regime separation after warm-up |

The simple nearest-centroid baseline shows the same clean transition at 0.50. At separation 0.35 Cortex sometimes begins to distinguish structure that the baseline keeps collapsed, but the representation is not clean: fragmentation and collisions remain substantial.

At high noise (0.18) Cortex sometimes stores one additional transient prototype even when post-warm-up assignments are perfect. This is a small but real structural-overhead signal.

### 2. Novelty detection is strong once novelty clears the boundary

| separation | noise | Cortex detection rate | median Cortex delay | baseline detection rate |
|---:|---:|---:|---:|---:|
| 0.35 | 0.03 | 3/3 | 7 observations | 0/3 |
| 0.35 | 0.12 | 1/3 | 3 observations among successful runs | 2/3 |
| 0.50 | 0.03-0.12 | 3/3 | 1 observation | 3/3 |
| 0.75+ | 0.03-0.12 | 3/3 | immediate | 3/3 |

Median false discoveries before the novelty event were zero in every tested novelty case.

The 0.35 band is the important weakness: small changes in noise move the system between detection and non-detection. This is a genuine phase boundary rather than a uniformly robust region.

### 3. Gradual drift exposes a fragmentation weakness

Cortex adapts cleanly through total RMS drift 0.40 in this fixture. Beyond that, the moving regime is increasingly represented as new structure rather than one continuously adapting identity.

| total drift | noise | Cortex target fragmentation | Cortex post-warm discoveries | baseline target fragmentation |
|---:|---:|---:|---:|---:|
| 0.20 | 0.03-0.12 | 1 | 0 | 1 |
| 0.40 | 0.03-0.12 | 1 | 0 | 1 |
| 0.80 | 0.03 | 2 | 1 | 2 |
| 0.80 | 0.12 | 3 | 1 | 2 |
| 1.20 | 0.03 | 3 | 2 | 2 |
| 1.20 | 0.12 | 4 | 2 | 2 |

Overall cluster purity remains close to 1 because the fragments are mostly pure. That is precisely why purity alone would hide this failure mode.

No split or merge promotions occurred in this particular drift fixture. The immediate research question is therefore not whether Cortex can distinguish the drifting observations, but whether its structural-plasticity policy should reconcile those fragments more aggressively when the drift is continuous.

### 4. Hard regime budgets remain explicit and exact

For budgets 2, 4, and 8, the fixture presents exactly three additional mutually separated regimes. Each overflow regime is repeated eight times.

| budget | attempted regimes | stored | budget-pressure observations | unresolved observations |
|---:|---:|---:|---:|---:|
| 2 | 5 | 2 | 24 | 24 |
| 4 | 7 | 4 | 24 | 24 |
| 8 | 11 | 8 | 24 | 24 |

Cortex never exceeds the declared budget. Every one of the 24 observations belonging to the three unrepresentable regimes is surfaced explicitly as budget pressure / unresolved rather than causing hidden state growth.

### 5. Full-stack binding is robust to Gaussian noise but brittle to sparse corruption

The identity fixture contains 12 true entities, six visible per frame, feature dimension 16, and quarter-cycle nuisance transformations. It runs for 700 frames with 100 frames excluded as warm-up.

With **no sparse corruption**, Cortex finishes all cases with:

- median binding purity `1.0`;
- median fragmentation `1.0`;
- exactly 12 active identities;
- the correct four-element nuisance subgroup;
- no capacity failure;

for additive noise from `0.01` through `0.22`.

Sparse feature corruption changes the picture substantially:

| noise | corruption probability | purity | fragmentation | active entities | capacity-failure rate |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.03 | 1.000 | 1.42 | 18 | 0/3 |
| 0.05 | 0.03 | 1.000 | 1.42 | 18 | 0/3 |
| 0.12 | 0.03 | 1.000 | 1.50 | 18 | 0/3 |
| 0.22 | 0.03 | 0.9997 | 1.83 | 22 | 0/3 |
| 0.01 | 0.10 | 0.9986 | 3.67 | 42 | 0/3 |
| 0.05 | 0.10 | 0.9983 | 3.75 | 41 | 0/3 |
| 0.12 | 0.10 | 0.9978 | 4.08 | 46 | 0/3 |
| 0.22 | 0.10 | 0.9966 | 5.75 | 64 | 2/3 |

This is the strongest weakness uncovered by `characterization-v1`: the binder usually avoids merging different true entities, but corrupted observations cause it to spawn duplicate identities. At the hardest tested point, two of three repetitions exhaust the 64-entity capacity even though there are only 12 true entities.

The nuisance subgroup remains correctly selected with a large positive margin throughout these cases, so the failure is localized more strongly to corrupted-observation binding/spawn behavior than to nuisance-group inference.

Simply increasing `max_entities` would hide the symptom while increasing matching cost. It would not solve the scientific problem. The better next target is corruption-aware binding / duplicate reconciliation, with an ablation showing whether reliability weighting, spawn gating, or later merge logic is responsible.

### 6. The continual core remains bounded and numerically finite over 8,000 steps

For noise 0.05 and 0.15:

- zero non-finite values were found in the final snapshot;
- stored regimes remained at 5 under budget 12;
- budget pressure and unresolved counts remained zero;
- roughly 663-666 reactivation events occurred;
- median mean-step latency was about 1.37-1.38 microseconds on the GitHub runner;
- median p95 was about 1.6-1.9 microseconds;
- median p99 was about 20 microseconds.

The stream has four nominal recurring regimes; the fifth stored regime is consistent with the reversible drift fixture producing an additional structural distinction. This suite establishes only finite 8,000-step continual-core stability; it is not a full-runtime endurance result.

## Current strength / weakness map

The tested system is strongest when latent regimes are separated beyond the configured recurrence boundary and when identity observations contain additive noise rather than sparse gross corruption. It also enforces structural budgets exactly and exposes overload explicitly.

The clearest weaknesses are:

1. **boundary ambiguity** around the recurrence/novelty tolerance;
2. **over-fragmentation under continuous large drift**;
3. **severe identity over-fragmentation under sparse feature corruption**, eventually causing explicit entity-capacity failure;
4. occasional transient extra prototypes under high noise even when steady-state assignments are correct.

These are useful failures because they are localized and measurable rather than generic score degradation.

## Non-claims and limitations

- The fixtures are synthetic and internally generated; they do not establish external validity.
- The nearest-centroid comparator is a sanity baseline, not a competitive neural or continual-learning baseline.
- Three seeds are enough to expose deterministic failure patterns here, but not enough for population-level statistical claims.
- The identity suite scores entity binding and nuisance inference; it does not yet score the relation/causal graph against ground truth.
- The long-run suite exercises the continual core, not the complete native runtime.
- No result here is a proof of a universal Cortex property. Structural invariants such as budget boundedness should be proved separately.

## Next discriminating experiments

The results suggest a focused next campaign rather than indiscriminate feature growth:

1. corruption-aware binding ablations to isolate spawn-gate, reliability, and duplicate-reconciliation effects;
2. full-stack relation and causal ground-truth scoring after identity mapping;
3. nuisance-group misspecification and changing nuisance groups;
4. longer full-runtime endurance with resident-memory tracking;
5. untouched external structured data with frozen configuration and stronger baselines.

The first item has the highest information value because `characterization-v1` found a reproducible transition from clean identity tracking to structural explosion while nuisance inference itself remains stable.
