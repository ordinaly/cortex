# Probe Sampling v0.4 — Results

## Outcome

**Pass.** The budgeted active/latent scheduler reduced scheduled distinction
queries by **78.63%** on the delayed noisy horizon-3 fixture while preserving
continuous active sampling and the declared latent revisit guarantee.

## Fixture

Probe Economy v0.3 had already established for this world:

```text
14 declared probes
2 active probes
12 latent probes
```

The v0.4 scheduler used:

```text
budget_per_tick = 3
latent_max_gap = 12
run length = 120 ticks
```

The active probes were declared-probe indices `[9, 13]`.

## Measured query load

| Measure | Result |
|---|---:|
| Uniform 14-probe baseline | 1,680 queries |
| Scheduled queries | 359 |
| Active queries | 240 |
| Latent sentinel queries | 119 |
| Relative query saving | 78.63% |
| Maximum observed latent revisit gap | 12 ticks |

Both active probes were queried on all 120 ticks. Every latent probe was queried
at least once, and every measured repeated latent interval remained within the
12-tick contract.

## Interpretation

The result establishes a limited but useful separation between representation
and observation cost:

```text
full declared question family
        |
        +-- small active basis -> dense sampling
        |
        +-- latent remainder   -> sparse bounded sentinel sampling
```

Cortex therefore no longer needs to spend one query per declared distinction on
every tick merely to keep dormant distinctions recoverable.

This is not yet autonomous rediscovery. v0.4 schedules sentinel observations
but does not maintain a dedicated recent latent statistic or escalate sampling
when a latent question starts carrying new information. That closed loop is the
next milestone.
