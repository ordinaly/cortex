# Closed Articulation Loop v0.1

Status: **research prototype / generic invariants passed / A-D development probe complete**.

This prototype completes, at the research layer, the missing feedback path between
foreground articulation, downward consolidation, memory, and later
re-articulation.

It does not modify spec-v0.9.5 or the native runtime contract.

## Motivation

The stable runtime currently computes one 12-dimensional articulation vector and
feeds that same vector independently to fuzzy historical memory and continual
reasoning. Historical memory reconstruction is deliberately not fed back into
the continual core.

That boundary was useful for semantic safety, but it leaves the runtime open-loop
with respect to memory:

    articulation -> memory

without the complementary path:

    memory -> conditioned future articulation

Open-World Robustness development stages A-D exposed four limits that all touch
this missing interaction: evidence dependence, temporal provenance, compound
background explanation, and bounded relevance under memory pressure.

The prototype therefore tests a generic closed loop rather than four stage-
specific fixes.

## Functional correspondence

The mechanism is intentionally named in implementation terms rather than
Sigma-Lambda ontology, but its functional decomposition is:

    incoming structured evidence
              |
              v
        foreground articulation
              |
              v
       downward consolidation
        /       |        \
 provenance   residual   temporal
 redundancy   structure  provenance
        \       |        /
              v
        bounded memory
        active / latent
              |
       conditioned activation
              |
       guarded bubbling
              |
              v
      future articulation

Memory is persistent state. Downward articulation is the operation that
consolidates, reorganizes, coarsens, and preserves residual structure. Upward
articulation remains the currently executable foreground.

## Generic evidence contract

The research loop consumes EvidenceAtom values containing only:

- an opaque context tuple;
- an opaque relation/value;
- event time;
- arrival time;
- opaque provenance;
- an optional confidence scalar.

No source truth, hidden reliability, novelty class, benchmark tag, or self-model
is present.

## Downward operations

### Repeated-articulation compression

Repeated context/value observations accumulate into one MemoryBasin rather than
one persistent object per event.

Each basin retains:

- effective evidence mass;
- raw count;
- utility;
- first/last event time;
- last arrival time;
- provenance set;
- active/latent state.

### Learner-visible provenance redundancy

The loop compares provenance streams only where they make claims about the same opaque context at the same local
observation ordinal. This makes the comparison insensitive to reporting order
while preserving per-context chronology.

Two provenance streams may enter the same dependence class when they have enough
overlap and sufficiently high repeated agreement.

Dependence does not by itself imply simultaneity. The loop separately learns
whether aligned multi-source reports arrive in compact temporal bursts separated
by materially larger gaps. Only that burst/separation geometry enables
episode-level foreground consolidation, preventing continuously alternating
reporters from being mistaken for one simultaneous evidence packet.

Once both provenance structure and episode structure are mature, downward
articulation operates on the whole context-local evidence episode before
foreground plasticity. Reports from
one inferred dependence class first collapse to one group vote; the remaining
independent group votes are then combined into one foreground articulation for
that ordinal. A tied episode stays unresolved rather than advancing the
foreground clock.

Raw reports still enter historical memory with fractional effective mass, so
the audit trail is preserved while neither copied evidence nor report ordering
gets multiple foreground votes.

This is compression of redundant evidence, not a learned truth score.

### Temporal provenance

For each context the loop maintains the greatest event time observed so far.

An observation arriving later with an earlier event time is still consolidated
into historical memory, but is marked historical-only and does not
automatically rewrite the current foreground articulation.

This separates:

    evidence belongs to history

from:

    evidence is authorized to alter the present

without deleting or falsifying the late observation.

### Residual/background structure

Novel coordinates and contradictions to the current foreground become generic
StructuralChange records.

Changes within a bounded window are combined into a BackgroundCandidate. A
candidate may contain several simultaneous changes. It is only marked
bubble-eligible after repeated multi-subject support; background structure does
not execute by itself.

## Memory and the accordion

Memory has two bounded tiers:

- active basins available for immediate conditioning;
- latent basins retained for possible later reactivation.

When active capacity is exceeded, low-utility distinctions are demoted.

When total capacity is exceeded, the least useful latent distinction is
coarsened into aggregate mass rather than raising an exception.

Thus the research semantics are:

    articulate
      -> compress
      -> demote
      -> reopen when relevant
      -> coarsen when distinction cost exceeds retained utility

The aggregate coarsened count and mass remain auditable even when the original
distinction is no longer represented.

## Conditioned activation

Memory lookup is context-conditioned. The whole memory is never injected into
foreground reasoning.

Relevant basins are ranked by retained evidence mass and decayed utility.
Activation raises their utility and may move them back into the active tier.

## Guarded recurrence bubbling

A historical basin cannot execute from memory alone.

Reactivation requires:

1. the relation was genuinely observed before;
2. sufficient historical effective mass remains;
3. fresh, non-late evidence now supports the same relation;
4. that fresh support persists for a configured confirmation streak.

When the gate passes, the existing regime tracker receives an opt-in
reactivate_from_memory transition.

That transition changes only current articulation state. It does not increment
history or fast evidence, so memory is never disguised as a new observation.

## Explicit non-goals

v0.1 does not:

- introduce a Cortex self-model;
- assign semantic meaning to context positions;
- know OWR stage labels;
- infer source reliability or truth;
- mutate the native runtime;
- solve native entity-budget pressure yet;
- authorize background candidates as beliefs;
- generate official OWR results.

The first scientific question is narrower:

> Does this generic closed-loop mechanism satisfy its own invariants and then
> improve more than one OWR failure without benchmark-specific rules?

The generic invariants passed before the A-D development adapter was evaluated.
The adapter uses one unchanged default policy across all four stages and
consumes only learner-visible observation fields during execution.

## Initial invariants

The test suite requires:

- repeated evidence compresses into one basin;
- late evidence remains historical without rewriting foreground state;
- repeated redundant provenance can be inferred from public agreement alone;
- recurrence can reactivate from real memory without fabricating history;
- simultaneous residuals remain grouped as a joint background candidate;
- coordinate novelty is exposed generically rather than mapped to hard labels;
- memory pressure coarsens instead of throwing;
- conditioned activation remains context-local.

These are mechanism invariants, not OWR performance gates.


## A-D development probe

The first integrated probe used 12 development worlds per OWR stage with one
unchanged ClosedLoopPolicy. Hidden oracle fields were used only after execution
for scoring. No official seed was generated and OWR-E remained untouched.

### OWR-A — evidence dependence

The frozen source-blind Cortex had previously preserved safety by resolving
0/48 copied-majority traps. The closed loop instead learned redundant
provenance structure from calibration history, consolidated mature evidence
episodes before foreground plasticity, and achieved:

| partition | coverage | resolved accuracy | selective risk |
|---|---:|---:|---:|
| all | 0.7500 | 0.9921 | 0.0079 |
| calibration | 0.7396 | 0.9859 | 0.0141 |
| fresh targets | 0.7639 | **1.0000** | **0.0000** |
| target controls | 0.9167 | **1.0000** | **0.0000** |
| copied-majority traps | **0.6875** | **1.0000** | **0.0000** |

Thus 33 of 48 trap queries were answered correctly and none incorrectly.
Coverage remains deliberately incomplete: unresolved evidence episodes do not
force a foreground relation.

### OWR-B — temporal disorder

The event-time frontier and guarded recurrence path substantially reduce the
unsafe transition behavior measured in the frozen system.

| partition | coverage | resolved accuracy | selective risk |
|---|---:|---:|---:|
| all | 0.5021 | 0.8983 | 0.1017 |
| near A -> B | 0.1094 | 0.3810 | 0.6190 |
| settled B | 0.7396 | **1.0000** | **0.0000** |
| near B -> C | 0.2604 | 0.7200 | 0.2800 |
| settled C | 0.8021 | 0.9870 | 0.0130 |
| near C -> A recurrence | 0.5833 | 0.8214 | 0.1786 |
| settled recurrence | 0.9063 | 0.9885 | 0.0115 |

The previous frozen selective risks near the three transition regions were
1.0000, 1.0000, and 0.9787 respectively. The generic loop therefore produces a
large improvement, but **OWR-B is not solved**: near the first transition,
resolved errors remain unacceptably common.

### OWR-C — compound background structure

After learning temporal episode geometry separately from provenance dependence,
the loop preserves the existing local relation result:

```text
local relation coverage  = 1.000
local relation accuracy  = 1.000
selective risk           = 0.000
```

Every one of the 12 worlds also produced generic multi-subject compound
background candidates. Pattern-selective bubbling reduced the earlier noisy
prototype behavior to two eligible compound candidate patterns per world.

The mechanism still does **not** emit benchmark semantic labels such as
`new_entity_type` or `new_relation`. The result therefore demonstrates
generic joint structural candidate formation, not completed OWR-C semantic
attribution.

### OWR-D — bounded research memory

With the same default loop, the 3,000-event development streams complete without
a research-layer capacity exception:

```text
hard failure rate       = 0.000
mean stored basins      = 256.0
mean coarsened basins   = 2254.17
query coverage          = 0.7756
resolved accuracy       = 1.000
selective risk          = 0.000
```

This is the intended accordion behavior at the research-memory layer: pressure
reduces represented distinctions and therefore coverage rather than producing
confidently wrong answers or unbounded growth.

It is **not yet a fix for the native entity-capacity cliff**. Native structured
knowledge ingestion still allocates physical entities before this research loop
can coarsen them. Promotion requires a native integration point where downward
memory decisions can influence representation allocation.

## Development conclusion

One generic closed-loop mechanism improves more than one independent failure
without introducing OWR stage labels:

```text
foreground articulation
        |
downward evidence / residual / temporal consolidation
        |
bounded active + latent memory
        |
conditioned activation
        |
guarded bubbling
        |
future articulation
```

The strongest evidence is currently OWR-A and the bounded research behavior in
OWR-D. OWR-B improves substantially but retains unsafe transition cases.
OWR-C gains a generic joint-candidate representation but not semantic component
attribution.

The next engineering question is therefore not another benchmark-specific
patch. It is whether the closed-loop semantics should be integrated deeper into
the native representation lifecycle while preserving the frozen runtime as the
differential reference.
