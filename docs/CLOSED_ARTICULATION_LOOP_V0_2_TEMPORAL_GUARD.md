# Closed Articulation Loop v0.2 — Temporal Disorder Guard

Status: **development refinement after static-policy confirmation failure**.

## Motivation

The static v0.2 confirmation panel did not promote any candidate.

The best static temporal candidate reduced OWR-B transition wrong-answer rate
substantially, but it also reduced OWR-D coverage beyond the frozen regression
limit. This indicates that globally increasing foreground sensitivity trades
transition safety against stable-stream availability.

The confirmation panel also isolated one safe improvement:

- lowering `recurrence_min_history` from 5 to 3 materially increased correct
  OWR-B transition coverage;
- it did not change OWR-A, OWR-C, or OWR-D on the confirmation material;
- by itself it did not sufficiently reduce wrong transition answers.

The next refinement therefore keeps the v0.1 foreground
`RegimePlasticityPolicy` unchanged and makes execution permission conditional
on observed temporal disorder.

## Mechanism

For each opaque context, downward temporal processing maintains a non-negative
disorder pressure.

For each arriving event:

```text
p <- decay * p

if event_time is older than the context frontier:
    p <- p + 1
```

Late observations remain historical-only exactly as in v0.1.

If an optional guard is enabled and:

```text
p >= threshold
```

then an otherwise executable foreground read is exposed as `UNRESOLVED`.

The guard:

- does not change the accepted relation;
- does not change durable history;
- does not change fast evidence;
- does not fabricate a contradiction;
- automatically decays away after in-order evidence resumes.

It is disabled by default until validation.

## Frozen tuning grid

The candidate starts from the v0.1 closed-loop policy with:

```text
recurrence_min_history = 3
recurrence_confirmations = 3
```

The global foreground regime policy remains v0.1.

Only two new guard parameters are swept:

```text
temporal_disorder_decay in
    {0.25, 0.50, 0.65, 0.75, 0.85, 0.90}

temporal_guard_threshold in
    {0.50, 0.75, 1.00, 1.25, 1.50, 2.00}
```

Eight OWR-B development worlds select a Pareto frontier minimizing transition
wrong-answer rate while maximizing transition correct-answer rate.

The three nearest frontier points to the two-objective ideal proceed to a
separate 20-world-per-stage validation material.

## Validation panel

The validation panel contains:

1. v0.1 baseline;
2. recurrence-only candidate;
3. the three frozen temporal-guard finalists.

A guard finalist may be promoted only if:

- OWR-A trap selective risk = 0;
- OWR-A trap coverage regresses by no more than 0.02 absolute;
- OWR-B transition wrong-answer rate <= 75% of baseline;
- OWR-B transition correct-answer rate >= baseline;
- every settled B regime has selective risk <= 0.03;
- OWR-C local coverage and resolved accuracy remain 1;
- every C world still emits a multi-subject candidate;
- OWR-D selective risk = 0;
- OWR-D coverage regresses by no more than 0.02 absolute.

If no candidate passes, v0.1 remains the default.

No official seed and no OWR-E world is used.
