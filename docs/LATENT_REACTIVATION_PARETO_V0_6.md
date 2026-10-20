# Latent Reactivation Pareto v0.6

## Status

Research-only calibration following Latent Probe Reactivation v0.5. No native
runtime semantics change.

## Motivation

The deterministic v0.5 reference passed its speedup gate narrowly:

```text
sparse-only reactivation: 17 ticks
escalated reactivation:   11 ticks
speedup:                   1.545x
```

but escalation spent more queries. That result is too close to the acceptance
threshold to justify fixing the current trigger parameters or immediately
building learned connectivity on top of them.

v0.6 therefore characterizes the attention tradeoff under stochastic answers.

## Confidence-support correction

During preparation we found a streaming-fidelity issue in learned Delta.
`TracePredictiveMemory` predicts from a bounded sliding window, but the learner's
confidence radius previously used a cumulative direct-observation counter. After
a window had discarded old samples, those discarded samples could still shrink
the confidence radius of the current empirical response.

v0.6 corrects this by exposing the retained support of the exact predictor row:

```text
n_q(x) = number of currently retained samples used by predict(x,q).
```

`EmpiricalTraceDistinguishability.support()` now returns this retained count.
This also correctly includes prefix samples contributed by longer trace probes
when those prefix samples actually drive the shorter-probe predictor.

The cumulative observation ledger remains only an accounting statistic.

## Noisy calibration world

Two opaque contexts and two one-step probes are used.

Before the regime switch:

```text
probe 0: P(1|a)=0.10, P(1|b)=0.90   # useful
probe 1: P(1|a)=0.20, P(1|b)=0.20   # redundant / latent
```

After the switch:

```text
probe 0: P(1|a)=0.20, P(1|b)=0.20   # becomes redundant
probe 1: P(1|a)=0.10, P(1|b)=0.90   # becomes useful
```

Hidden phase labels are evaluator-only. Cortex receives only sampled probe
outcomes.

The learned predictive memory uses a 32-sample window and empirical merge
threshold 0.35 for this attention-policy calibration. Confidence is disabled in
the calibration fixture so the experiment isolates sentinel-trigger behavior;
the retained-support correction is tested independently.

## Parameter grid

The development grid spans:

```text
latent sentinel gap: 3, 5 ticks
sentinel window:      3, 5, 7 samples
trigger threshold:   0.40, 0.55, 0.70 TV
escalation budget:   8, 16 dense samples
```

for 36 candidate policies.

Each candidate is evaluated across fixed development seeds with:

- 60 stable ticks before the change;
- up to 100 post-change ticks;
- false-trigger rate;
- false activation of the latent probe during the stable regime;
- reactivation success rate;
- mean reactivation delay;
- speedup relative to sparse-only sampling at the same sentinel gap;
- mean query saving relative to uniform two-probe sampling;
- mean number of priority queries.

## Pareto rule

A candidate dominates another only if it is no worse in all of:

```text
false-trigger rate       lower is better
false-activation rate    lower is better
reactivation delay       lower is better
query saving             higher is better
reactivation rate        higher is better
```

and strictly better in at least one.

A candidate is considered safe enough for this development stage when:

```text
false-trigger rate    <= 0.10
false-activation rate <= 0.10
reactivation rate     >= 0.90
speedup vs sparse     > 1.0
query saving          > 0
```

The benchmark passes only if both a non-empty Pareto frontier and at least one
such safe candidate exist.

## Boundary

This is a development-grid calibration, not a proof of optimality or statistical
consistency. The next decision should depend on the measured frontier. If no
candidate survives, the correct outcome is to revise or abandon the v0.5
escalation mechanism before attempting learned connectivity.
