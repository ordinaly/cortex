# Probe Economy v0.3

## Status

Research-only extension of the learned predictive distinguishability line. It does
not change the frozen native runtime or `spec-v0.9.5`.

## Goal

Previous milestones showed that Cortex can learn a fixed bank of predictive
trace distinguishers from deterministic and stochastic evidence. The remaining
artificial ingredient is that every declared probe stays permanently active.

Probe Economy v0.3 asks a narrower question:

> Given the distinctions currently justified by the full learned probe family,
> what is the smallest probe population needed to preserve them?

The finite Sigma-Lambda oracle is **not** used to choose probes. It remains an
independent evaluator after reduction.

## Merge-veto relation

Let `P` be the current declared probe family and `X` the current context set. For
each learned probe `q`, Cortex already has an empirical total-variation response
and a confidence radius.

For a pair `(x,y)`, probe `q` vetoes a merge when

```text
empirical_TV_q(x,y) + radius_q(x) + radius_q(y) > epsilon.
```

Insufficiently supported pairs also veto merging. This preserves the existing
rule that lack of evidence is not evidence of equivalence.

Define the empirical veto-edge set

```text
S_q = {(x,y) : q vetoes merging x and y}.
```

The full learned distinction population therefore imposes

```text
S = union_q S_q.
```

A reduced probe population `A` is sufficient exactly when

```text
union_{q in A} S_q = S.
```

For the bounded research setting, minimum probe selection is a finite minimum
set-cover problem:

```text
minimize |A|
subject to union_{q in A} S_q = S.
```

The implementation solves this exactly for at most 18 probes. This limit is a
research resource guard, not a theoretical bound.

## Marginal utility

For an active probe `q`, its empirical distinction contribution is the fraction
of currently required veto edges that would become uncovered if `q` were
removed. For a latent probe, its gain is the fraction of required edges it would
newly cover if activated.

A normalized diagnostic utility is

```text
U_q = marginal_distinction_gain_q - probe_cost.
```

The constant per-probe cost is deliberately simple in v0.3. Later work may
replace it with measured memory, sampling, or compute cost.

The hard sufficiency constraint dominates utility: Cortex never drops a probe
merely because the representation cost is high if doing so would erase a
currently required empirical distinction.

## Hysteresis

`HystereticProbePopulation` maintains active and latent probes.

- If the current active set no longer covers the full empirical distinction
  relation, missing probes are activated immediately.
- If the active set is sufficient, a global basis swap is considered only when
  it reduces population size by at least the activation threshold.
- Active probes are retired only when full empirical coverage remains intact.
- Among equally small exact covers, the solver maximizes overlap with the
  currently active population before applying declared probe order.

The final rule is the main persistence mechanism: equivalent minimal bases do
not churn simply because several choices exist.

## Experiments

The v0.3 benchmark applies the unchanged learned-Delta mechanism to:

1. deterministic `action_blindness`, horizon 2;
2. noisy immediate controlled world, horizon 2;
3. noisy delayed controlled world, horizon 3.

For each case:

- train the full probe bank from learner-visible action/output traces;
- compute the empirical minimum probe cover without hidden state;
- reduce the active population;
- require the reduced empirical quotient to equal the full empirical quotient;
- audit that reduced quotient against the independent exact oracle;
- require the active population to match the exact empirical minimum cardinality
  and be strictly smaller than the declared full bank.

A separate regime-switch unit test moves the useful distinction from one
one-step probe to another. The first probe must become latent and the second must
reactivate while preserving full empirical coverage.

## Interpretation boundary

A probe being redundant here means only that it is redundant **for the current
learned merge relation on the current context set**. It does not mean the
corresponding question is globally meaningless. Latent probes remain available
for reactivation after regime change.

Likewise, minimum set cover is not claimed to be the final biological or
cognitive mechanism. It is an exact finite reference for testing approximate
future controllers.

## Next step

If v0.3 passes, the next milestone should stop assuming a complete declared probe
family is sampled uniformly. The controller should allocate observation effort
among active and latent questions, then test whether useful probes can be
rediscovered under bounded sampling cost before moving on to learned neuronal
connectivity.
