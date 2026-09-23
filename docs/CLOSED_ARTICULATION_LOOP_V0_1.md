# Closed Articulation Loop v0.1

Status: **research prototype / generic invariants under validation**.

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

The loop compares provenance streams only where they make claims about the same
opaque context at the same event time.

Two provenance streams may enter the same dependence class when they have enough
overlap and sufficiently high repeated agreement.

A dependence class shares one unit of current evidence weight. This is
compression of redundant evidence, not a learned truth score.

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

Only after the generic invariants pass should adapters be added for A-D
development worlds.

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
