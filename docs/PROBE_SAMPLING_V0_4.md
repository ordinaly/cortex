# Probe Sampling v0.4

## Status

Research-only extension of Probe Economy v0.3. It does not modify the frozen
native runtime or `spec-v0.9.5`.

## Goal

Probe Economy v0.3 established that Cortex can compress a learned distinction
population to a much smaller active basis while preserving the empirical
quotient. However, all declared probes were still sampled uniformly.

Probe Sampling v0.4 introduces a separate resource policy:

> sample active questions continuously, but revisit latent questions often
> enough that they can become relevant again.

This preserves the distinction between **representation economy** and
**attention/sampling economy**.

## Scheduler

Let `A_t` be the active probe set and `L_t` the latent set at tick `t`.

The v0.4 reference scheduler imposes:

```text
active period = 1 tick
latent period <= G ticks
queries per tick <= B
```

where `G` is `latent_max_gap` and `B` is `budget_per_tick`.

The policy is accepted only if

```text
|A_t| <= B
```

and there is enough remaining service capacity to revisit every latent probe:

```text
|L_t| <= (B - |A_t|) * G.
```

If this is not satisfied, the scheduler raises an error instead of silently
starving latent questions.

## Deadline rule

Active probes are queried every tick.

Latent probes are assigned staggered sentinel deadlines across the available
latent query slots. After a latent probe is sampled, its next deadline is

```text
current_tick + latent_max_gap.
```

A latent probe that remains latent keeps this deadline; synchronization with the
active population does not repeatedly pull it forward. If the active basis
changes, newly latent probes receive fresh staggered deadlines while newly
active probes immediately enter the every-tick schedule.

## Why this is separate from probe utility

A probe can be representationally redundant now and still be scientifically
useful later. Therefore:

```text
latent != deleted
```

v0.3 controls whether a probe contributes to the current distinction geometry.
v0.4 controls how often Cortex spends evidence-gathering effort on it.

This gives a first operational form of attention with protected exploration:

```text
current relevance -> dense sampling
latent possibility -> sparse sentinel sampling
```

## Benchmark

The reference benchmark uses the noisy delayed horizon-3 world from v0.2/v0.3.
That fixture has:

```text
14 declared probes
2 empirically necessary active probes
12 latent probes
```

The preregistered scheduler uses:

```text
budget_per_tick = 3
latent_max_gap = 12
```

so both active probes can be queried every tick and the remaining slot can
service one latent sentinel. Twelve latent probes can therefore all be revisited
within the declared 12-tick bound.

The benchmark runs for 120 ticks and requires:

- every active probe selected on every tick;
- every latent probe selected at least once;
- observed latent revisit gaps no greater than 12 ticks;
- no tick exceeding the three-query budget;
- at least 75% fewer scheduled probe queries than uniform 14-probe sampling.

## Current boundary

This milestone schedules latent sentinel queries but does not yet use sentinel
outcomes to trigger probe reactivation autonomously. The existing v0.3
population controller can already reactivate a probe once its learned evidence
changes, but v0.4 does not yet connect sparse sentinel evidence to a dedicated
fast reactivation statistic.

The next step should therefore add a small recent sentinel model for latent
probes and test the complete loop:

```text
latent probe
  -> sparse sentinel evidence
  -> change detected
  -> temporary sampling escalation
  -> full learner refresh
  -> probe reactivation if still useful
```

Only after that loop is validated should Cortex attempt to invent new questions
or learn neuronal connectivity.
