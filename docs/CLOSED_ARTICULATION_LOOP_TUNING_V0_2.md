# Closed Articulation Loop v0.2 — Tuning Protocol

Status: **development tuning preregistration**.

This campaign fine-tunes the research-only Closed Articulation Loop v0.1
without changing the OWR generators, using OWR-E, generating official seeds, or
modifying the native runtime.

## Principle

The loop has several parameters with different functional roles. A brute-force
Cartesian sweep would be expensive and would make it easy to compensate one
failure with an unrelated knob.

The campaign therefore tunes four mechanism groups separately:

1. evidence/provenance episode geometry on OWR-A;
2. foreground temporal plasticity and recurrence on OWR-B;
3. residual/background bubbling on OWR-C;
4. accordion memory pressure on OWR-D.

Only the top two candidates from each final factor group are recombined. This
produces 16 joint candidates rather than a large unconstrained grid.

## Data separation

Two independently derived development materials are used:

```text
closed-loop-v0.2-tuning-<stage>
closed-loop-v0.2-validation-<stage>
```

The first material selects local candidates and the joint Pareto frontier.

The second material is a holdout used only after the joint candidates are
frozen. It contains 12 worlds per stage by default.

These are development worlds, not official OWR worlds.

OWR-E remains untouched.

## Frozen search ranges

### A — evidence geometry

```text
dependence_min_overlap   in {4, 6, 8}
dependence_agreement     in {0.85, 0.875, 0.90}
episode_min_completed    in {2, 3}
episode_separation_ratio in {1.5, 2.0}
```

Primary safety requirements:

- copied-majority trap selective risk = 0;
- target-control selective risk = 0;
- calibration selective risk <= 0.05.

Among feasible candidates, higher trap coverage is preferred first.

### B — foreground temporal core

The global Relational Regime Plasticity v0.1 defaults are not changed. The
closed loop may, however, use a different local RegimePlasticityPolicy if the
holdout validates it.

Core search:

```text
fast_decay         in {0.35, 0.50, 0.65}
pressure_decay     in {0.85, 0.90}
suspicion_ratio    in {0.03, 0.05, 0.08}
confirmation_steps in {3, 5}
maturity_cap       = 64
```

The top core candidates then receive a recurrence sweep:

```text
recurrence_min_history  in {3, 5, 8}
recurrence_confirmations in {2, 3, 4}
```

Primary objective: reduce the wrong-answer rate across all near-transition
queries while preserving useful correct coverage.

Settled-regime selective risk must remain <= 0.03.

### C — background residual articulation

```text
residual_window      in {3, 4, 6, 8, 10}
compound_min_events  in {2, 3, 4, 5}
```

Requirements:

- local relation coverage = 1;
- local relation resolved accuracy = 1;
- every development world produces at least one multi-subject compound
  candidate.

Among feasible candidates, the smallest excess candidate count is preferred.
This discourages background overproduction.

### D — accordion memory

```text
total basin budget  in {128, 192, 256, 384}
active fraction     in {0.25, 0.50}
utility_decay       in {0.995, 0.997, 0.999}
```

Requirements:

- no research-layer hard failure;
- selective risk = 0.

The local memory frontier jointly minimizes budget and maximizes query coverage
and accuracy. Both the highest-coverage and highest-efficiency points may
advance to joint evaluation.

## Joint feasibility gates

A recombined candidate is feasible only if:

- OWR-A trap risk = 0;
- OWR-A target risk <= 0.01;
- OWR-C local coverage and resolved accuracy = 1;
- OWR-C forms a multi-subject candidate in every world;
- OWR-D hard failure rate = 0;
- OWR-D selective risk = 0;
- all three settled OWR-B regimes have selective risk <= 0.03.

## Pareto objectives

Feasible joint candidates are compared without a single weighted performance
score on:

```text
maximize A trap coverage
minimize B transition wrong rate
maximize B transition correct rate
maximize D query coverage
minimize total memory budget
```

Non-dominated candidates form the tuning Pareto frontier.

For the sole purpose of selecting up to three holdout finalists, equal-weight
normalized Euclidean distance to the observed ideal point is used across those
five objectives.

The same rule is then applied to the holdout finalists. If no finalist satisfies
all frozen feasibility gates, the campaign records failure and does not promote
a new default.

## Promotion boundary

A validated v0.2 parameter set may become the default **inside
ClosedArticulationLoop only**.

This campaign must not:

- change the global RegimePlasticityPolicy defaults;
- alter spec-v0.9.5;
- alter the native runtime;
- generate official OWR seeds;
- inspect or tune on OWR-E;
- reinterpret an abstention as a correct answer.

Negative results are retained.
