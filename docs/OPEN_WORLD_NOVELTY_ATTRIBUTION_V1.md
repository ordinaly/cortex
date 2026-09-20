# Cortex Reasoning Map v1 — Open-World Novelty Attribution v1

Status: **passed official benchmark / research-only**.

Protocol: `open-world-novelty-attribution-v1`.

Official run: GitHub Actions run `35511592906`, job `106080323461`.

## Question

When new evidence is poorly explained by Cortex's active world model, can the
system distinguish **what kind of structural change is justified** instead of
merely emitting a generic anomaly signal?

The frozen attribution space was:

- noise / no persistent edit;
- new instance of an existing entity type;
- new entity type;
- new relation;
- new predictive regime;
- unresolved when competing structural explanations cannot be separated.

## Official campaign

Untouched official seeds **1500–1519** across eight structural families
produced **160 cases**.

Every frozen gate passed.

| metric | result |
|---|---:|
| exact attribution accuracy | **1.000** |
| resolved/unresolved correctness | **1.000** |
| pure non-mutating evaluation | **1.000** |
| exact edit-locality rate | **1.000** |
| required persistent edit rate | **1.000** |
| post-edit no-further-change stability | **1.000** |
| false persistent edit rate | **0.000** |
| resolved score-margin correctness | **1.000** |
| ambiguous-case unresolved rate | **1.000** |
| transient recovery/no-edit correctness | **1.000** |
| nearest-known-type baseline accuracy | **0.375** |
| global-reset baseline accuracy | **0.125** |

## Model-selection mechanism

Cortex compares predictive-description explanations rather than directly
classifying an observation.

Each candidate receives a score consisting of:

- data negative log-likelihood;
- structural edit cost.

The best candidate is accepted only when it clears the frozen margin over the
second-best candidate.

This makes the unresolved state operational: ambiguity is represented as a
failure to justify one model edit over another.

## Noise versus structural novelty

Both noise families were left structurally unchanged in every official case.

The result is important because a novelty mechanism that always modifies the
ontology would be unsafe for continual reasoning.

The same conservative behavior held for the transient-excursion control: the
small temporary deviation and the later recovery evidence both remained
compatible with no persistent edit.

This control tests **non-persistence under a weak temporary excursion**. It
does not yet test discovering, tracking, and later reactivating a genuinely
strong temporary regime.

## New instance

An unseen entity whose predictive behavior matched a known type was assimilated
by changing only the instance-to-type binding.

No new type, relation, or regime was created.

Edit-locality correctness: **1.000**.

## New entity type

When two unseen entities shared a predictive profile strongly incompatible with
all existing types, Cortex introduced one new pooled predictive type and bound
only those unseen entities to it.

The edit changed exactly:

- type library;
- new-instance bindings.

No unrelated ontology component changed.

## New relation

Repeated evidence for an absent typed relation schema produced a relation-only
edit.

Known entity identity, entity types, and active regime remained unchanged.

## New regime

Coherent predictive change across all six known entities was attributed to a
new regime rather than to new entity identity or new entity types.

The edit changed exactly:

- regime library;
- active regime pointer.

Entity ontology and relation structure remained intact.

## Ambiguous instance versus type

The intentionally ambiguous fixture was calibrated so that:

- assimilating one unseen entity as a known type;
- creating a new entity type;

were separated by less than the frozen score margin.

Cortex returned **unresolved** in every official case.

The top two explanations were exactly:

- `new_instance`;
- `new_entity_type`.

No edit was emitted.

This is the strongest qualitative result of the milestone: the system can
represent **uncertainty about the kind of ontology change required**, rather
than forcing a novelty label.

## Post-edit stability

For every resolved structural novelty case, applying the proposed edit made the
same evidence subsequently resolve as **noise / no further structural change**.

This passed in all official cases.

That gives the edit an operational interpretation:

> the proposed structural change is sufficient to explain the residual evidence
> without immediately demanding another structural change.

## Baseline discrimination

The nearest-known-type baseline reached **0.375** exact attribution accuracy.

It systematically over-assimilated unseen entities and could not correctly
separate new type, new relation, and ambiguity cases.

The global-reset baseline reached **0.125**.

It treated most sufficiently large residuals as regime changes.

The point is not that these are state-of-the-art baselines. They test whether
Cortex is doing more than the two simplest pathological strategies:

- always assimilate;
- always reset globally.

## Architectural consequence

The Reasoning Map now contains the chain:

```text
predictive residual
        |
competing structural explanations
        |
noise / instance / type / relation / regime / unresolved
        |
localized model edit
```

Combined with proof dependency and planning/replanning, this creates the basis
for a stronger integrated loop:

```text
surprise
-> novelty attribution
-> model edit
-> dependency invalidation
-> local or global plan repair
```

That integration has **not** yet been benchmarked in this milestone.

## Claim boundary

A defensible claim is:

> Within declared finite structured worlds, Cortex can compare multiple
> predictive-description explanations for surprising evidence, distinguish
> noise from new instances, new entity types, new relations, and new regimes,
> preserve unresolved state when two structural explanations are
> indistinguishable, and localize the persistent model edit required by a
> resolved attribution.

This result does **not** establish:

- arbitrary real-world open-set recognition;
- raw sensory novelty detection;
- adversarial robustness;
- simultaneous compound novelty;
- calibrated probabilistic novelty confidence;
- autonomous scientific discovery;
- scalable open-ended ontology growth.

## Reproducibility

Implementation:

- `research/open_world_novelty.py`
- `research/open_world_novelty_campaign.py`
- `research/open_world_novelty_gate.py`
- `tests/test_open_world_novelty.py`

Frozen protocol:

- [OPEN_WORLD_NOVELTY_ATTRIBUTION_V1_PROTOCOL.md](OPEN_WORLD_NOVELTY_ATTRIBUTION_V1_PROTOCOL.md)

Aggregate result:

- [`benchmarks/results/open_world_novelty_attribution_v1_summary.json`](../benchmarks/results/open_world_novelty_attribution_v1_summary.json)

## Recommended next milestone

The strongest immediate follow-up is **Novelty-Aware Replanning v1**.

The experiment should test whether different novelty attributions trigger
different scopes of revision:

- noise -> preserve the current plan;
- new instance -> bind locally and continue when possible;
- new relation -> invalidate only relation-dependent plan support;
- new entity type -> revise ontology-dependent plan support;
- new regime -> reconsider transition assumptions more broadly;
- unresolved -> defer irreversible structural revision.

A parallel robustness campaign should then add stochastic corruption,
persistence sweeps, compound novelty, and calibration curves.
