# Open-World Robustness v1 — OWR-A development characterization

Status: **development characterization complete / no Cortex mechanism change**.

Protocol: \`open-world-robustness-v1-owr-a-development\`.

Hosted development run: GitHub Actions run \`35693585153\`.

## Question

When source reliability and source dependence are hidden, can current Cortex
transfer what it learned about sources on earlier contexts to fresh target
contexts?

The revised fixture uses eight calibration contexts followed by six previously
unseen target contexts. Four target contexts contain a coherent copied-majority
trap: one weak source and two correlated copies oppose two independent strong
sources.

No source reliability, copy graph, truth label, or oracle field is supplied to
Cortex.

## Why the first fixture was rejected

The first development fixture introduced misinformation only after a long
history on the same target contexts.

Current Cortex became conservative, but cumulative majority still solved the
final targets from historical inertia. That fixture therefore did not isolate
transferable source epistemics and was rejected before any official run.

The revised fixture removes prior target history entirely.

## Development result

Twelve development worlds produced:

- 96 calibration queries;
- 72 fresh-target queries;
- 48 copied-majority trap queries;
- 24 transfer-control queries.

### Current Cortex

Current Phase-B temporal Cortex was run with every hidden source reliability
forced to unit weight.

| partition | accuracy | coverage | resolved accuracy | selective risk |
|---|---:|---:|---:|---:|
| calibration | 0.542 | 0.542 | 1.000 | 0.000 |
| all fresh targets | 0.278 | 0.278 | 1.000 | 0.000 |
| target controls | 0.833 | 0.833 | 1.000 | 0.000 |
| copied-majority traps | **0.000** | **0.000** | n/a | **0.000** |

The trap result is a conservative failure: Cortex does not recover the target
truth, but it also does not execute the wrong copied-majority relation.

Across the complete characterization, every Cortex-resolved query was correct.

## Baselines

### Context-local cumulative majority

| partition | accuracy | coverage | selective risk |
|---|---:|---:|---:|
| calibration | 1.000 | 1.000 | 0.000 |
| fresh targets | 0.333 | 1.000 | 0.667 |
| target controls | 1.000 | 1.000 | 0.000 |
| copied-majority traps | **0.000** | **1.000** | **1.000** |

The revised fixture therefore closes the historical-majority escape hatch.

### Recent-five vote

On copied-majority traps:

- accuracy: **0.0625**;
- coverage: **0.9792**;
- selective risk: **0.9362**.

Naive recency is actively unsafe in this fixture.

### Public dependence-collapsed baseline

A simple baseline estimates unusually high pairwise source agreement on the
calibration stream, collapses highly dependent sources into one vote group, and
then transfers that grouping to fresh targets.

It receives no hidden reliability values, copy graph, target truth, or oracle
labels.

Result:

- calibration accuracy / coverage: **1.000 / 1.000**;
- target accuracy / coverage: **1.000 / 1.000**;
- copied-majority trap accuracy / coverage: **1.000 / 1.000**.

This is important because it demonstrates that the learner-visible stream
contains sufficient dependence information. The task is not solvable only by
an evaluator oracle.

### Oracle independence-aware reference

The evaluator-only reference also achieved **1.000 accuracy / 1.000 coverage**
on every partition.

## Interpretation

OWR-A exposes a specific missing capability rather than a generic accuracy
failure.

Current Cortex:

- recognizes enough conflict to withhold execution;
- preserves zero resolved error in the declared development worlds;
- does not infer a transferable source-dependence model;
- therefore cannot recover truth on fresh contexts where correlated sources
  create a false numerical majority.

The result supports a narrow next research question:

> Can source identity and cross-context agreement become revisable relational
> evidence inside Cortex without turning source reputation into a hard,
> permanent predicate?

No source-learning mechanism is introduced in this milestone yet.

## Gate status

Performance gates remain **unfrozen**.

This is development characterization only. No official OWR-A seed has been
generated.

The next campaign step is to characterize OWR-B through OWR-D with the current
frozen Cortex before deciding which missing mechanisms should be developed.
