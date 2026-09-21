# Graph Sandwich Routing v0.1

Status: **exploratory research campaign active**.

This experiment asks whether the cost of exact relational articulation can be
predicted cheaply enough to skip the current full routing profile on some
layered execution graphs.

It does **not** change Cortex semantics.

## Motivation

The current articulation controller profiles an admissible relation signature
exactly before selecting native Cortex traversal or the LPU. The exact profile
is already substantially better than complete path enumeration because it
merges identical layered states, but it still scans every reachable state at
the selected resolution and materializes an exact structural profile.

The research question is therefore narrower than path-count acceleration:

> Can a bounded local structural probe identify clearly cheap and clearly
> combinatorial execution regimes before exact profiling?

## Theorem boundary

The inspiration is the Kim--Vu sandwich conjecture, proved in full by Natalie
Behague, Daniel Il'kovic and Richard Montgomery (arXiv:2510.20765). It states,
at a high level, that a random d-regular graph in the regime
d = omega(log n) can be coupled between nearby Erdos--Renyi graphs.

Related earlier work on random biregular bipartite graphs is especially
suggestive for layered execution structures:
Klimošová, Reiher, Ruciński and Šileikis,
"Sandwiching biregular random graphs" (arXiv:2010.15751).

Neither theorem applies automatically to Cortex.

Cortex layered graphs are learned, finite, resolution-filtered, often small,
and can be highly clustered or adversarial. Degree balance is not sufficient
to establish the hypotheses of either random-graph result.

Accordingly, v0.1 is explicitly **theorem-inspired rather than
theorem-certified**.

## Architecture

The experiment adds a read-only probe:

```text
Cortex relational hypergeometry
          |
          v
admissible composition rule
          |
          +------> bounded degree probe
          |             |
          |             +--> clear cheap envelope --> Cortex
          |             |
          |             +--> clear combinatorial envelope --> LPU
          |             |
          |             +--> irregular / overlapping --> exact fallback
          |
          +------> existing exact profiler
```

The probe cannot alter:

- learned rules;
- relation confidence;
- evidence resolution;
- accepted memory;
- candidate answers;
- witness validity.

A wrong routing estimate can therefore only change computational cost.

## Probe statistics

For each sampled layer the probe records:

- observed frontier size;
- sampled state count;
- local matching degree;
- coefficient of variation of degree;
- observed next-state count;
- whether the frontier was truncated.

The branch-factor envelope uses a conservative relative slack based on both a
fixed floor and measured degree variation.

The probe selects LPU only when the lower path and lower compression envelopes
both exceed the existing routing thresholds. It selects native Cortex when the
upper envelope lies wholly below at least one threshold. All overlapping or
visibly irregular cases fall back to the exact profiler.

These are engineering envelopes, not statistical confidence intervals.

## Deterministic control

The campaign also includes a saturating exact profile.

It caps large path-count arithmetic while retaining exact structural traversal.
This is an important control: if the deterministic mechanism achieves the same
speedup, the sandwich-inspired estimator is unnecessary complexity.

The current expectation is that saturation will reduce integer growth but will
not reduce neighborhood scans.

## Graph families

The v0.1 campaign includes:

1. regular reconverging layered lattices;
2. near-regular lattices;
3. equal-degree clustered structures;
4. visibly irregular structures;
5. an anti-sampling adversarial family whose low-degree states are deliberately
   aligned with the deterministic probe positions.

The adversarial family is intentional. It tests whether apparent local
regularity can hide a globally different routing regime.

## Metrics

Primary metrics:

- direct routing coverage;
- direct route agreement with exact profiling;
- exact-fallback rate;
- false Cortex routes;
- false LPU routes;
- matching-neighborhood queries;
- exact-profile / probe wall-time ratio;
- semantic equivalence of Cortex and LPU execution on manageable fixtures.

## Promotion rule

v0.1 is not promoted merely because it is faster on regular lattices.

A useful result requires:

1. exact execution semantics remain unchanged;
2. the estimator materially reduces structural work on wide regular or
   near-regular frontiers;
3. estimator overhead is lower than the exact profile it replaces;
4. irregular structures mostly abstain or misroute rarely enough that expected
   end-to-end cost improves;
5. the gain remains after comparison with the simpler deterministic saturation
   control.

If these conditions do not hold, the experiment should be recorded as a
negative result and the extra routing complexity should not enter Cortex.

## Sigma--Lambda interpretation

The probe is a temporary articulation-cost estimate. It has no authority over
the underlying relational field. In Sigma--Lambda language, it is an
engineering aid to Lambda-up articulation, not a change to the Sigma domain or
durable Lambda-down memory.
