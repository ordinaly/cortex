# Possibility Geometry v0.1 — finite reference layer

Status: implemented opt-in research oracle, 21 October 2026. The frozen native
runtime contract is unchanged. See the [formal proposal](CORTEX_POSSIBILITY_GEOMETRY_FORMAL_V0_1.md).

## What this milestone implements

`research/possibility_geometry.py` provides a finite model of mutually exclusive
situated alternatives, each with references to caller-owned shared memory and
local assumptions. References are opaque identifiers, not copied memory or a
validated native-memory adapter. The caller supplies the candidates and model.

The joint kernel specifies successor and observation probabilities for each
candidate and action. `PossibilityFilter.predict(action)` freezes the prior,
per-candidate observation laws, predicted successor mass, and joint predictive
law before `assimilate(action, observation)` conditions it. Only one prediction
can be pending. A zero-likelihood event returns `model_failure`, retains the
observation and forecast, and blocks further updates until external revision
creates a new filter. The retained old belief is diagnostic, not a posterior
at the failed step. An unknown observation also yields explicit failure.

All probabilities use exact rational arithmetic. Use `Fraction` or decimal
strings for intended exact values; a Python float is interpreted as its exact
binary rational value. Inputs must normalize exactly; invalid kernels are
rejected rather than repaired silently. This is a correctness oracle, not a
high-throughput inference implementation.

`trace_distribution` computes finite-horizon observable trace probabilities.
A `Probe` specifies an open-loop action sequence and an accepted set of traces.
Its answer probability is derived from the model, ensuring compatible probes
share one generative law. Adaptive observation-conditioned policies are deferred.

`geometry` computes the maximum probe disagreement between each pair. It stores
the flag complex implicitly using vertices and edges; `contains` checks simplex
membership without enumerating all cliques. Probability mass remains on vertices.
Reweighting alone does not change this geometry. Connected similarity chains are
not treated as simplices or merged classes.

`is_exact_partition` checks equality of joint successor-cell/observation mass
for every action among all members of each proposed cell. It does not execute
compression. This is stronger than equality on the selected finite probes.

## Existing Cortex integration boundary

The existing `GeometricTemporalTransport.kernel` normalizes exponentiated
similarity scores. Those are useful compatibility scores, but are not by
construction calibrated environment transitions. This oracle therefore does not
reuse them as a generative kernel. A future adapter must estimate and validate
transitions from chronological observations and separately define the observation
mechanism. It must also specify how native context/entity IDs become mutually
exclusive candidate states; overlapping verbal causes cannot simply be normalized.

The new filter retains only its current belief, one pending forecast, or one
failure record. The caller owns audit-history retention. Candidate count and
probe horizons are finite and caller-chosen, not autonomously budgeted. Trace
enumeration can grow exponentially in horizon, and exact rational arithmetic can
be expensive. No scaling or resource advantage is claimed.

## Validation and example

From the repository root:

```bash
python -m pytest tests/test_possibility_geometry.py
PYTHONPATH=research python research/possibility_geometry_demo.py
```

Acceptance coverage: survival preserves two cup explanations; inspection resolves
them; rare evidence remains a soft posterior; equal likelihood preserves prior
odds; impossible and out-of-alphabet observations retain failure witnesses;
predictions cannot be overwritten or consumed twice; predecessor-dependent
observations use the general joint law; multistep forecasts match exhaustive
hidden-path enumeration; joint traces separate equal one-step marginals;
similarity does not restrict temporal reachability; nontransitive similarity
cannot authorize merging; and exact partition validation has positive and negative
fixtures. Explicitly modelled missing observations use their declared mechanism.

The CI Python correctness job includes these tests. The demo is deterministic
and uses no hidden simulator information beyond its supplied model.

## Deferred gates

Learning candidates, probes and likelihoods; native-memory/controller wiring;
automatic revision; pruning and recurrence recovery; actual quotient construction;
approximate compression with measured retained cost; action selection; calibration
on held-out streams; and matched-budget comparisons remain future milestones.
Passing finite semantics tests does not establish autonomous learning or AGI.

For the research rationale and source boundaries, see
[Semantics for Possibility Geometry](CORTEX_POSSIBILITY_SEMANTICS.md).
