# Latent Probe Reactivation v0.5 — Results

## Outcome

**Pass, narrowly.** Sentinel-triggered dense sampling materially accelerated
reactivation in the deterministic regime-switch reference, but the gain came at
an explicit query-cost penalty.

| Measure | Sparse-only | Sentinel escalation |
|---|---:|---:|
| Stable false trigger | no | no |
| Drift trigger after change | tick 5 | tick 5 |
| Reactivation after change | tick 17 | tick 11 |
| Final active basis | `((1,),)` | `((1,),)` |
| Priority queries | 0 | 6 |
| Total queries through reactivation | 39 | 40 |
| Uniform baseline through reactivation | 68 | 56 |
| Relative query saving | 42.65% | 28.57% |

Measured reactivation-latency speedup:

```text
17 / 11 = 1.545x
```

The preregistered gate required at least 1.5x, so the candidate passes by a
small margin.

## Interpretation

The important result is not that escalation is universally better. It is that
sparse sentinel evidence can close the full loop:

```text
latent question
  -> sentinel drift
  -> observation priority
  -> refreshed learned Delta
  -> changed empirical distinction relation
  -> Probe Economy reactivation
```

The monitor itself never changed the active representation. It triggered once,
used six priority queries, and the ordinary economy ultimately replaced the old
active probe `(0,)` with the newly useful probe `(1,)`.

The sparse-only arm also recovered, because v0.4 guarantees latent revisits. It
simply required six additional post-change ticks. Therefore escalation trades
sampling economy for adaptation speed rather than creating a capability that
was otherwise impossible.

## Caution

This first result is deterministic and the speedup is close to the acceptance
threshold. It does not justify freezing the current trigger/window/escalation
parameters as a research default.

Before moving to learned neuron-neuron connectivity, the next experiment should
characterize the Pareto surface over:

- false-trigger rate under stationary noise;
- reactivation delay under real regime change;
- total query cost;
- escalation duration and sentinel revisit interval.

The current mechanism is retained as a reference point, not an optimized
attention policy.
