# Cortex Reasoning Map v1 — Open-World Novelty Attribution v1

Status: **frozen protocol / official campaign passed**.

Protocol ID: `open-world-novelty-attribution-v1`.

## Research question

When current evidence is poorly explained by Cortex's active world model, can
Cortex distinguish **what kind of model change is justified**, rather than
merely flagging an anomaly?

The frozen attribution space is:

- noise / no persistent edit;
- new instance of an existing entity type;
- new entity type;
- new relation;
- new predictive regime;
- unresolved when the best structural explanations are not distinguishable.

## Model-selection principle

Novelty attribution is treated as competing predictive-description
explanations.

For each hypothesis `H`, Cortex evaluates

```math
L(H) = L_data(H) + K(H)
```

where:

- `L_data` is the negative log-likelihood of the observed outcomes and
  relation evidence under the proposed model;
- `K(H)` is the declared structural edit cost.

The selected explanation is accepted only when its score is separated from the
second-best candidate by the frozen resolution margin.

Near-ties remain **unresolved** and produce no edit.

## Base world

Each official fixture begins with:

- two known entity types with distinct categorical predictive distributions;
- three known instances of each type;
- one known relation schema;
- no alternate regime active.

Type IDs, entity IDs, and relation IDs are opaque and independently randomized
per seed.

## Candidate explanations

### Noise

No structural change.

Known entities keep their active type probabilities. Unseen instances, if any,
are evaluated under the current global type mixture.

Complexity cost: **0**.

### New instance

Every unseen entity is bound to the existing type that minimizes its predictive
negative log-likelihood.

Complexity cost:

```math
K_instance = 2 x number_of_new_instances
```

No type, relation, or regime structure may change.

### New entity type

All unseen entities in the evidence window are pooled into one candidate latent
type with a smoothed categorical maximum-likelihood distribution.

Complexity cost:

```math
K_type = 5 + 2 x number_of_new_instances
```

The edit may change only:

- the type library;
- bindings for the newly observed instances.

### New relation

For an observed relation schema absent from the current typed relation model,
Cortex fits one smoothed Bernoulli relation probability.

Complexity cost: **4**.

The edit may change only the relation model.

### New regime

If evidence is available for at least three known instances, Cortex may fit
alternate categorical outcome distributions pooled by existing entity type.

Complexity cost:

```math
K_regime = 8 + 2 x number_of_affected_types
```

The edit may change only:

- the regime library;
- the active regime pointer.

Entity types, entity identity, and relation structure must remain unchanged.

## Frozen numerical constants

- categorical smoothing alpha: **0.5**;
- Bernoulli smoothing alpha: **0.5**;
- resolution margin: **0.5** score units;
- minimum known instances for a regime candidate: **3**;
- default unseen relation probability: **0.05**.

These constants are frozen before development evaluation.

## Benchmark families

Eight structural families are frozen.

### noise-spike

One known entity produces a small anomalous fluctuation that remains compatible
with its existing predictive type.

Expected attribution: **noise**.

Expected persistent edit: **none**.

### incoherent-noise

All six known entities contain modest deviations, but the deviations remain
close to their original type distributions and do not justify a coherent
alternate regime.

Expected attribution: **noise**.

Expected persistent edit: **none**.

### new-instance

One unseen entity has a predictive outcome profile matching one existing type.

Expected attribution: **new_instance**.

Expected edit locality: instance binding only.

### new-entity-type

Two unseen entities share a predictive outcome profile that is strongly
incompatible with every existing type.

Expected attribution: **new_entity_type**.

Expected edit locality: add one type and bind only the unseen entities.

### new-relation

Two known entities repeatedly express a relation schema absent from the current
typed relation model.

Expected attribution: **new_relation**.

Expected edit locality: relation model only.

### new-regime

All six known entities change behavior coherently while retaining their
existing entity identities and types.

Expected attribution: **new_regime**.

Expected edit locality: regime library + active regime only.

### transient-excursion

Known entities show a small temporary excursion followed by evidence consistent
with the original model.

Expected attribution for both the excursion window and recovery control:
**noise / no persistent edit**.

This family tests conservative revision, not temporary-regime discovery.

### ambiguous-instance-type

One unseen entity is calibrated so that:

- binding it to the nearest known type;
- introducing a new entity type;

have scores separated by less than the frozen resolution margin.

Expected attribution: **unresolved**.

Required behavior:

- the top two candidates are `new_instance` and `new_entity_type`;
- no persistent model edit is emitted.

## Persistent edit semantics

Attribution itself is pure: evaluating evidence must not mutate the model.

A structural proposal is applied only after a resolved attribution.

After applying a correct persistent edit, the same evidence must be explained
as **noise / no further structural change**.

Expected changed components:

| family | changed components |
|---|---|
| noise-spike | none |
| incoherent-noise | none |
| new-instance | instances |
| new-entity-type | instances + types |
| new-relation | relations |
| new-regime | regimes + active regime |
| transient-excursion | none |
| ambiguous-instance-type | none |

## Baselines

Two simple baselines are frozen.

### Nearest-known-type baseline

- unseen entities are always assimilated as new instances;
- broad known-entity drift is treated as a regime change;
- relation-only novelty is not structurally attributed.

### Global-reset baseline

Any evidence window with sufficient mass is treated as a global regime change.

These are not intended as state-of-the-art anomaly detectors. They test whether
the multi-hypothesis structural attribution is doing more than always
assimilating or always resetting.

## Development and official seeds

Development seeds:

```text
0 ... 9
```

Untouched official seeds:

```text
1500 ... 1519
```

With eight families, the official campaign contains **160 cases**.

The official seed range must not run until the development campaign and frozen
gates pass unchanged.

## Frozen gates

The campaign passes only if all of the following hold.

### Attribution

1. exact category attribution across all identifiable cases = **1.000**;
2. correct resolved/unresolved state = **1.000**;
3. every resolved case clears the **0.5** score margin;
4. ambiguous-instance-type unresolved rate = **1.000**;
5. ambiguous top-two competitor identity correctness = **1.000**.

### Model-edit discipline

6. evaluation purity = **1.000**;
7. exact edit-locality rate = **1.000**;
8. required persistent edit rate = **1.000**;
9. false persistent edit rate on noise/transient/ambiguous cases = **0**;
10. post-edit no-further-change stability = **1.000**;
11. transient recovery/no-edit correctness = **1.000**.

### Baseline discrimination

12. nearest-known-type baseline exact attribution accuracy <= **0.50**;
13. global-reset baseline exact attribution accuracy <= **0.25**.

No threshold may be weakened after official results are observed.

## Negative-result policy

If development finds an implementation or fixture error, it may be corrected
before any official run while preserving the scientific requirements above.

If the untouched official campaign fails, preserve the result. Any revised
official claim must use a new protocol version and fresh untouched seeds.

## Claim boundary

A passing result would support the narrow claim:

> Within declared finite structured worlds, Cortex can compare multiple
> predictive-description explanations for surprising evidence, distinguish
> noise from new instances, new entity types, new relations, and new regimes,
> preserve unresolved state when two structural explanations are
> indistinguishable, and localize the persistent model edit required by a
> resolved attribution.

It would not establish:

- arbitrary real-world open-set recognition;
- raw perceptual novelty detection;
- autonomous scientific discovery;
- robust attribution under adversarial noise;
- calibrated probabilistic novelty confidence;
- simultaneous compound novelty;
- large-scale ontology growth.

Those belong to later robustness and integration campaigns.

## Planned files

- `research/open_world_novelty.py`
- `research/open_world_novelty_campaign.py`
- `research/open_world_novelty_gate.py`
- `tests/test_open_world_novelty.py`

The official campaign must remain untouched until development preflight is
green.


## Official outcome

The untouched official campaign on seeds **1500–1519** passed every frozen
gate across **160** cases.

Official metrics:

- exact attribution accuracy: **1.000**;
- resolved/unresolved correctness: **1.000**;
- evaluation purity: **1.000**;
- exact edit locality: **1.000**;
- required persistent edit rate: **1.000**;
- false persistent edit rate: **0.000**;
- post-edit stability: **1.000**;
- ambiguous unresolved rate: **1.000**;
- transient recovery/no-edit correctness: **1.000**;
- nearest-known-type baseline accuracy: **0.375**;
- global-reset baseline accuracy: **0.125**.

See [OPEN_WORLD_NOVELTY_ATTRIBUTION_V1.md](OPEN_WORLD_NOVELTY_ATTRIBUTION_V1.md)
and
[`benchmarks/results/open_world_novelty_attribution_v1_summary.json`](../benchmarks/results/open_world_novelty_attribution_v1_summary.json).

The frozen gates and scoring constants above were not changed after official
evaluation.
