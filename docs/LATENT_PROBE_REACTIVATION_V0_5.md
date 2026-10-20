# Latent Probe Reactivation v0.5

## Status

Research-only closure of the Probe Economy v0.3 / Probe Sampling v0.4 loop. It
does not change `spec-v0.9.5` or the native runtime.

## Goal

v0.3 can make redundant questions latent. v0.4 can revisit them sparsely without
paying uniform sampling cost. v0.5 asks whether those sparse sentinel answers can
cause a dormant distinction to regain observation priority and then earn its way
back into the active representation.

The authority order is deliberately constrained:

```text
latent probe
  -> sparse sentinel answers
  -> short-window drift suspicion
  -> temporary sampling escalation
  -> ordinary distinguishability learner refresh
  -> ordinary Probe Economy decision
  -> reactivation only if the refreshed probe is structurally needed
```

The drift monitor therefore controls **attention**, not representation. It cannot
activate a probe directly.

## Latent baseline

When probe `q` becomes latent, the monitor freezes its current learned predictive
response for each monitored context:

```text
b_q(x) = learned response distribution at latency entry.
```

Sentinel observations are accumulated in a separate bounded recent window. For a
context with enough recent samples, the monitor computes empirical total
variation from the frozen baseline. The probe-level drift score is the maximum
across monitored contexts:

```text
R_q = max_x TV(recent_q(x), b_q(x)).
```

If

```text
R_q >= trigger_threshold,
```

then `q` receives a finite escalation budget. The baseline and short sentinel
window are diagnostic only; the main predictive representation remains owned by
the existing learned-Delta memory.

## Escalation

The v0.4 scheduler gains an optional `priority_latent` input.

- active probes remain non-preemptible;
- priority probes must still be latent;
- priority probes consume only the existing latent-capacity portion of the query
  budget;
- asking for more priority probes than the available capacity raises an error;
- ordinary v0.4 behavior is unchanged when no priority list is supplied.

Every escalated observation is also fed through the ordinary learned
Distinguishability mechanism. `HystereticProbePopulation` is reevaluated from
that refreshed evidence. If the probe becomes necessary, the normal economy
activates it; if not, escalation eventually expires and the refreshed response
becomes the new latent baseline.

This prevents a suspicious but ultimately redundant question from permanently
capturing attention.

## Reference regime-switch experiment

The first v0.5 fixture contains two one-step probes and two opaque contexts.
Initially:

```text
probe (0,) separates a from b
probe (1,) is redundant
```

Probe Economy therefore keeps `(0,)` active and makes `(1,)` latent.

After the regime changes:

```text
probe (0,) becomes redundant
probe (1,) becomes the separating question
```

The experiment compares two arms under the same finite learner windows:

1. **sparse-only:** latent evidence arrives only on the v0.4 sentinel schedule;
2. **sentinel escalation:** recent sentinel drift may temporarily promote the
   latent probe to dense sampling.

Both arms continue to use the ordinary learner and ordinary economy for final
reactivation.

Acceptance requires:

- no escalation trigger during a stable pre-change period;
- a finite post-change drift trigger;
- eventual active basis `((1,),)`;
- exactly one monitor-recorded reactivation;
- no query-budget violation;
- escalated reactivation strictly faster than sparse-only reactivation;
- at least 1.5x reactivation-latency speedup;
- positive query saving relative to fully uniform sampling.

## Boundaries

This is still a declared finite question family. v0.5 does not invent new
questions or learn neuron-neuron connectivity.

The reference detector also compares a short recent empirical window to a frozen
latency baseline. It is an engineering mechanism, not a theorem of optimal
change detection. Future work should test noisy false-trigger rates, tune the
attention/latency/query-cost Pareto surface, and only then consider learned
connectivity between surviving distinguishers.
