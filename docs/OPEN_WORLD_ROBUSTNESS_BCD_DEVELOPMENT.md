# Open-World Robustness v1 — OWR-B/C/D development characterization

Status: **development characterization complete / no Cortex mechanism change**.

Parent protocol: `open-world-robustness-v1`.

Hosted development run: GitHub Actions run `35824518020`.

This report records the development-only characterization of the frozen Cortex
system on OWR-B, OWR-C, and OWR-D. No official seed was generated, no
performance gate was frozen, and no Cortex mechanism was added or tuned against
these results.

OWR-A is reported separately in
[OPEN_WORLD_ROBUSTNESS_A_DEVELOPMENT.md](OPEN_WORLD_ROBUSTNESS_A_DEVELOPMENT.md).

## Executive summary

The three stages expose three distinct limits:

- **OWR-B:** Cortex is safe once a regime settles, but during delayed-evidence
  collision windows it can execute the stale previous regime with very high
  selective risk. Event-time replay and duplicate collapse alone do not remove
  this failure.
- **OWR-C:** Cortex tracks the local current relation perfectly in the revised
  fixture, but the current temporal surface cannot emit a compound structural
  explanation such as "new entity type + new relation schema".
- **OWR-D:** the unbounded temporal research surface remains accurate under the
  long distractor stream, while the actual bounded native runtime reaches a
  hard capacity failure at every tested capacity. Graceful degradation is not
  yet demonstrated.

Together with OWR-A, the development failure map is now:

```text
A  evidence dependence / source structure
B  temporal provenance / transition uncertainty
C  compound explanatory structure
D  bounded relevance / memory pressure
```

The common theme is that Cortex is stronger at representing relations in the
external world than at representing higher-order structure over its own
evidence, provenance, and active memory.

---

## OWR-B — Temporal Disorder and Recurrence

### Question

Can Cortex maintain a safe current articulation when event time and arrival time
differ, observations may be missing or duplicated, and old delayed evidence can
arrive after a genuine regime transition?

### Rejected development fixture v0

The first OWR-B development fixture queried only well inside stable regimes.

That fixture was rejected before any official run because a trivial recent-7
vote achieved perfect accuracy and coverage. It therefore did not test the
intended temporal problem.

### Development fixture v2 — boundary collision

The accepted development fixture concentrates long delays in the final eight
event-time steps before each regime boundary and increases observable duplicate
traffic. The latent regime remains

```text
A -> B -> C -> A
```

but queries now occur both inside the stale-evidence collision windows and after
the new regime has settled.

Across 12 worlds there were 960 queries.

### Frozen Cortex result

Current Cortex consumes evidence in arrival order. Hidden truth is never passed
to it and source reliability is fixed to unit weight for this stage.

| partition | coverage | resolved accuracy | selective risk |
|---|---:|---:|---:|
| all | 0.4052 | 0.6812 | 0.3188 |
| pre-shift | 0.8646 | **1.0000** | 0.0000 |
| near A -> B | 0.1979 | **0.0000** | **1.0000** |
| settled B | 0.4375 | **1.0000** | 0.0000 |
| near B -> C | 0.2083 | **0.0000** | **1.0000** |
| settled C | 0.5729 | **1.0000** | 0.0000 |
| near C -> A recurrence | 0.2448 | 0.0213 | **0.9787** |
| settled recurrence | 0.8750 | **1.0000** | 0.0000 |

The failure is therefore strongly localized in time.

When the regime has stabilized, every resolved Cortex answer in the declared
checkpoints is correct. During a transition window, however, a non-trivial
subset of contexts remains executable under the stale previous articulation.

This is not merely low coverage. The resolved answers near the first two
boundaries are wrong in every resolved case.

### Public temporal controls

Two public-metadata controls replay only observations that have actually
arrived by query time:

1. replay those arrived observations by their public event timestamp;
2. do the same after collapsing explicit duplicates.

Neither removes the transition failure.

For example, event-time replay near A -> B still has:

```text
coverage = 0.1667
resolved accuracy = 0.0000
selective risk = 1.0000
```

and the deduplicated version is identical on that partition.

A recent-7 arrival vote has much higher coverage, but is unsafe near
transitions:

```text
near A -> B selective risk = 0.6868
near B -> C selective risk = 0.6613
near recurrence selective risk = 0.6796
```

The deduplicated recent vote is similarly unsafe.

The development conclusion is therefore narrower than "Cortex ignores
timestamps." Event time and duplicate metadata are useful but not sufficient by
themselves. The missing behavior is better described as **transition-aware
temporal epistemics**: stale evidence must lose permission to sustain exact
execution when the evidence stream itself indicates that the world may have
crossed a regime boundary.

### Recurrence result

Mean arrival-time switch latency across the 12 worlds was:

```text
new B at t=45:       24.23
new C at t=85:       23.44
recurrent A at t=125: 23.74
```

The recurrence latency is not measurably better than the two novel regime
switches in this development characterization.

Thus the current durable history does not yet provide a demonstrated
reactivation advantage under this kind of temporal disorder.

### OWR-B development conclusion

OWR-B exposes a real unsafe-execution boundary:

> Cortex can recover correct stable-regime articulation, but it does not yet
> represent enough uncertainty about the temporal provenance of conflicting
> evidence to reliably suspend stale execution during regime transitions.

No fix is introduced here.

---

## OWR-C — Compound Novelty

### Question

Can Cortex explain an observation by more than one simultaneous structural
change while preserving ambiguity when several explanations remain equally
plausible?

### Development fixture correction before evaluation

The original Phase-1 fixture reused one opaque symbol both as a current relation
value and as evidence for a new relation schema. That made "new relation" and
"new regime" partially generator-defined rather than independently observable.

The fixture was corrected before any official run.

The revised fixture uses contexts of the form

```text
(entity, relation-schema)
```

and keeps the observed current outcome separate.

It contains:

- a coherent regime shift on known entities under a known relation schema;
- two unseen entities exhibiting the same behavior under a genuinely unseen
  relation schema;
- one unseen entity whose observations are deliberately compatible with either
  a new entity type or membership in the changed regime.

### Frozen Cortex result

Across 12 development worlds:

```text
local relation coverage  = 1.000
local relation accuracy  = 1.000
```

So the existing temporal Cortex surface correctly represents the current local
relation in every answerable case.

However:

```text
compound structural component output supported = false
compound structural component coverage          = 0.000
```

The current surface has no output vocabulary for a statement such as:

```text
{new_entity_type, new_relation}
```

It can represent the changed local relation, but not articulate the joint
structural explanation.

### Public-data discriminability control

A deliberately simple public-data structural baseline receives no oracle truth,
hidden tags, or expected component set.

On the revised development worlds it achieved:

```text
resolved compound exact rate   = 1.000
ambiguous unresolved rate      = 1.000
```

This control is not proposed as the Cortex solution. Its purpose is only to
show that the learner-visible stream contains enough information to distinguish
the declared resolved and unresolved cases.

### OWR-C development conclusion

OWR-C does not expose a failure of local relation tracking.

It exposes a missing **higher-order explanatory articulation**:

> Cortex can say what relation currently holds, but its current temporal
> interface cannot say which combination of structural changes best explains
> why the world now looks different.

No compound-attribution mechanism is added here.

---

## OWR-D — Memory and Interference

### Question

Under a long stream dominated by irrelevant contexts, can bounded Cortex memory
preserve important long-lived structure and degrade safely when capacity
pressure becomes severe?

### Development stream

The development world contains 3,000 events and hundreds of distractor
contexts. Only about 3.5% of events belong to salient long-lived contexts.

One salient context temporarily changes regime and later returns to its original
relation.

This stage deliberately compares two different surfaces:

1. the unbounded Python-side temporal regime bank, used as a cognitive
   reference;
2. the actual bounded native knowledge runtime, where `max_entities` is a real
   physical capacity constraint.

### Unbounded temporal reference

On the hosted development world:

```text
context count             = 332
active regime cells       = 2684
coverage                  = 0.9231
resolved accuracy         = 1.0000
recurrence correct        = 1.0000
```

The temporal mechanism itself therefore remains accurate on the declared
salient queries when physical entity capacity is not the limiting factor.

### Bounded native runtime

The native runtime was swept at three capacities:

| max entities | hard failure rate | stream processed | query coverage at failure | resolved accuracy |
|---:|---:|---:|---:|---:|
| 512 | **1.000** | 0.032 | 0.1538 | 1.0000 |
| 2,048 | **1.000** | 0.148 | 0.6923 | 0.8889 |
| 8,192 | **1.000** | 0.652 | 0.6154 | 1.0000 |

At every tested capacity the stream eventually ends in a hard runtime
exception rather than a declared memory-pressure state that safely reduces
coverage.

The recurrence query was not recovered in any bounded run because no tested
capacity completed the stream.

### Important limitation

The hosted OWR-D characterization currently contains **one full native world**.
The result is therefore a capacity-boundary confirmation, not a statistical
estimate of failure probability across random worlds.

This is sufficient to establish that graceful degradation is not yet guaranteed
by the current native knowledge-ingestion path. It is not sufficient to claim a
precise scaling law.

Before any official OWR-D campaign, the native sweep should be made cheap enough
to run across multiple development worlds and the relevant memory/state metrics
should be frozen.

### OWR-D development conclusion

The failure is not gradual accuracy degradation:

> the current structured-knowledge path continues allocating active
> representation until the configured entity capacity is exhausted, after
> which the stream fails instead of compressing, evicting, coarsening, or
> explicitly abstaining.

No memory-pressure policy is added here.

---

## Cross-stage failure map after OWR-A through OWR-D

The development investigation now exposes four limitations without modifying
Cortex.

### A — evidence-source structure

Cortex detects copied-majority conflict conservatively but cannot learn and
transfer source dependence across contexts.

### B — evidence-time structure

Cortex recovers after regime change, but stale prior evidence can retain exact
execution authority during temporally disordered transition windows.

### C — explanatory structure

Cortex correctly tracks local state but has no current mechanism for emitting
multiple simultaneous structural causes.

### D — relevance / bounded-state structure

The unbounded temporal mechanism remains accurate, but the bounded native
knowledge path has a hard capacity cliff rather than graceful representational
coarsening.

These are different measured failures, so they should not be collapsed into one
mechanism prematurely.

However, they share a useful architectural question:

> Can Cortex represent and revise **relations over its own evidence** —
> dependence, temporal provenance, explanatory coupling, and relevance — using
> the same plastic relational principles it already applies to the external
> world?

That question should guide the design phase, but this characterization does not
yet prescribe an implementation.

---

## Gate status

Performance gates for OWR-A through OWR-D remain **unfrozen**.

No official seed has been generated.

OWR-E remains untouched and must not be used for iterative development.

The next step is architectural synthesis from the A-D failure map, followed by
development-only mechanism work. Only after the candidate behavior, metrics,
and gates are frozen should official A-D worlds and the one-shot OWR-E composite
exam be generated.
