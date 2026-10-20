# Distinction Topology Robustness v0.8

## Question

v0.7 established that learner-visible mutual complementarity can induce a pairwise graph over surviving distinction neurons. v0.8 asks whether that graph is temporally usable:

1. does stationary observation noise cause edge chatter?
2. when the phenomenon really changes, does the graph rewire promptly?
3. does hysteresis improve stability without freezing obsolete topology?

This milestone remains pairwise. It does not introduce triangles or higher simplices.

## Three-neuron noisy world

The benchmark keeps three probe neurons alive over four opaque contexts `a,b,c,d`. Probe answers are Bernoulli samples. The learner receives only those samples.

The response probabilities are deliberately close enough to the learner's merge threshold to make the learned veto relation fluctuate under a finite sliding window.

Before the regime switch:

- probe 0 implements `a | bcd`;
- probe 1 implements `ab | cd`;
- probe 2 implements `abc | d`.

With the v0.7 complementarity rule, the evaluator-level ideal scores are:

- Gamma(0,1) = 1/6;
- Gamma(0,2) = 1/3;
- Gamma(1,2) = 1/6.

Thus a reference activation threshold near 0.30 yields one structural edge: `(0,2)`.

At the regime switch only probe 2 changes, becoming `ac | bd`. The ideal scores become:

- Gamma(0,1) = 1/6;
- Gamma(0,2) = 1/6;
- Gamma(1,2) = 1/3.

The same three vertices therefore need to rewire from `(0,2)` to `(1,2)`.

The hidden Bernoulli laws and these ideal targets are evaluator-only. They are not available to Probe Economy or Distinction Connectivity.

## Temporal hysteresis

An absent edge is created when

```text
Gamma >= edge_on
```

and an existing edge is removed when

```text
Gamma < edge_off.
```

with `edge_off <= edge_on`.

The gap is a temporal regularizer on relation evidence. It creates three qualitatively different regimes:

- **too little hysteresis** (`edge_off` close to `edge_on`): noisy scores repeatedly create/delete edges;
- **too much hysteresis** (`edge_off` very low): stale edges survive a real phenomenon change, producing topological lock-in;
- **intermediate hysteresis**: suppresses finite-window noise while still releasing obsolete relations.

This is directly analogous to Cortex plasticity elsewhere: stability and adaptability are competing objectives, not independently maximizable properties.

## Calibration protocol

The learner uses:

- horizon 1;
- three declared probes;
- four contexts;
- sliding window 64;
- epsilon 0.35;
- confidence disabled for this stress test, so finite-window sampling noise directly perturbs learned merge-veto sets.

Each seed receives:

1. a 64-round warmup under the initial regime;
2. 100 measured stationary ticks;
3. a regime switch;
4. 120 measured adaptation ticks.

The first 20 stationary ticks are excluded from the stability score. Successful rewiring means the post-switch target topology is held for five consecutive ticks.

Development seeds are `0..11`; holdout seeds are `12..23`.

The calibration grid scans:

- `edge_on` in `{0.20, 0.25, 0.30, 0.35}`;
- `edge_off` in `{0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35}` with `edge_off <= edge_on`.

A development policy is admissible only if it satisfies all of:

- stationary topology error <= 10%;
- mean stationary flips <= 3;
- rewiring success >= 95%;
- mean rewiring delay <= one 64-sample window;
- late post-switch topology error <= 15%.

Among admissible policies, selection prioritizes stationary correctness, post-switch correctness, low churn, then latency. Exact ties prefer the stricter activation threshold.

## Falsification criteria

v0.8 fails if no nonzero hysteresis band survives development, or if the selected policy fails holdout requirements:

- stationary error <= 12%;
- mean stationary flips <= 3;
- rewiring success >= 95%;
- mean rewiring delay <= 64 ticks;
- late post-switch error <= 15%.

It also fails if the selected band cannot beat a no-hysteresis baseline on late post-switch correctness, or if an intentionally sticky baseline does not exhibit worse plasticity. Those comparisons are included to ensure that the benchmark actually distinguishes the two failure modes rather than merely rewarding one arbitrary threshold pair.

## Scope

The result, if positive, establishes only that the v0.7 pairwise graph can be temporally stabilized on a controlled noisy streaming problem. It does not show global optimality, universal calibration, or justify higher-order simplices. The next topology milestone should be attempted only after this gate passes.
