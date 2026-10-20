# Learned Distinguishability v0.1

Status: **research experiment / evaluator oracle remains isolated**.

This stage replaces the oracle-supplied predictive `Delta` neurons from Five-Operator Bench v0.1 with responses learned from observable action/output traces.

The finite Sigma-Lambda oracle remains evaluator-only. `python/cortex/learned_distinguishability.py` does not import `research/sl_bench` or hidden world state.

## Learning contract

The learner receives only:

- an opaque context identifier;
- a declared finite action probe;
- the output word actually observed under that probe.

For each probe it learns an empirical output-word distribution. Pairwise probe distinguishability is empirical total variation.

Population distinguishability is:

```text
Delta(x, y) = sup_probe TV(P_hat(. | x, probe), P_hat(. | y, probe))
```

This uses the Sigma-Lambda accessible-question convention directly: the population is as distinguishing as its strongest accessible probe.

## Conservative quotient proposal

A proposed fold is allowed only when every declared probe has enough support and:

```text
empirical_TV + radius(x) + radius(y) <= epsilon
```

for every pair joined by complete-link clustering.

Unsupported contexts therefore remain separate. Confidence uncertainty cannot be used as evidence of equivalence.

The confidence radius is a bounded Hoeffding-style engineering guard over the declared finite context/probe budget. The exact oracle remains responsible for deciding whether a proposed quotient is actually valid in the benchmark world.

## First experiment

The first experiment uses SL-Bench's deterministic `action_blindness` world.

At horizon 1, observable emissions cannot distinguish states 0, 1 and 2, so the exact predictive quotient is:

```text
((0, 1, 2), (3,))
```

At horizon 2, action-conditioned temporal evidence reveals state 1 while states 0 and 2 remain equivalent:

```text
((0, 2), (1,), (3,))
```

The learner sees sampled trials only. After training, its empirical complete-link quotient is compared against the independently computed exact quotient and `audit_partition` certificate.

This tests a central intended behavior:

> articulation should refine when a newly accessible distinction becomes supported, not because a semantic label was inserted by the programmer.

## Five-operator integration

`EmpiricalTraceDistinguishability.build_bank()` exposes each learned probe response as a `DistinctionNeuron`. This means learned predictive distinctions can already feed the shared `Delta/K/C/V/I` architecture without changing the five-operator loop.

The neuron bank and the quotient proposal serve different purposes:

- neuron responses carry graded operational distinguishability into Cortex;
- the confidence-adjusted complete-link proposal decides when empirical evidence is strong enough to suggest a quotient fold.

## Non-claims

v0.1 still assumes the finite action-probe family is declared in advance. It does not yet:

- discover useful probes/actions;
- learn cross-channel structural or causal distinguishers;
- learn the potential neuronal simplicial complex;
- optimize the number of neurons;
- establish population guarantees under arbitrary nonstationary streams;
- change the frozen native runtime.

## Next stage

If the deterministic delayed/action distinction is recovered faithfully, the next experiment should move to noisy controlled worlds and measure:

- oracle quotient distortion;
- excess cell count over the minimum epsilon-feasible frontier;
- held-out predictive loss;
- active probe count;
- topology churn;
- drift/recurrence recovery.

After that, candidate probes can compete on predictive gain versus complexity so that the distinguishability population itself becomes plastic.
