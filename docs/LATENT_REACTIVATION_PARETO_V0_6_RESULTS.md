# Latent Reactivation Pareto v0.6 — Results

## Outcome

**Pass.** The noisy development grid contains a non-empty Pareto frontier and
multiple policies that satisfy the preregistered safety/adaptation criteria.

The main scientific result is that the v0.5 escalation mechanism is not merely
a deterministic artifact: under Bernoulli probe noise, several policies showed
zero observed false triggers across the 12 development seeds while materially
accelerating reactivation relative to sparse-only sentinel sampling.

This is development evidence, not a bound on future false-trigger probability.

## Retained-support correctness fix

Calibration also exposed a separate fidelity issue in learned Delta.
Predictions use bounded sliding-window trace memory, while confidence had been
computed from a cumulative historical observation count. Old samples could
therefore continue shrinking the confidence radius even after they no longer
contributed to the current predictor.

v0.6 corrects this:

```text
statistical support = retained samples that actually determine predict(x,q)
```

Longer-probe observations count toward a shorter prefix probe exactly when their
stored prefix samples contribute to that shorter predictor. Cumulative direct
observations remain accounting-only.

Focused regression tests verify that a predictor with window 8 has support 8
after 40 direct observations, not 40.

## Calibration grid

36 policies were evaluated:

```text
latent gap:          3, 5 ticks
sentinel window:     3, 5, 7 samples
TV trigger:          0.40, 0.55, 0.70
escalation budget:   8, 16 samples
seeds:               0..11
```

All 36 candidates met the minimum reactivation eligibility criterion on this
development fixture. The frontier displays a clear tradeoff between adaptation
speed, query cost, and false-trigger exposure.

## Safety-first reference policy

The recommendation rule was deliberately ordered by false-trigger safety before
raw speed. The selected development reference is:

```text
latent gap       = 5
sentinel window  = 7
TV threshold     = 0.40
escalation       = 16
```

Measured across 12 development seeds:

| Measure | Result |
|---|---:|
| Observed false-trigger rate | 0 / 12 |
| Observed false-activation rate | 0 / 12 |
| Reactivation rate | 12 / 12 |
| Sparse-only mean reactivation delay | 66.0 ticks |
| Escalated mean reactivation delay | 26.92 ticks |
| Mean speedup vs sparse | 2.45x |
| Mean query saving vs uniform two-probe sampling | 39.22% |
| Mean priority queries | 10.08 |

Again, `0 / 12` is an observed development result, not evidence that the true
false-trigger probability is zero.

## Representative frontier points

The frontier contains useful alternatives rather than one universally dominant
setting.

| Gap | Window | Threshold | Escalation | False triggers | Mean delay | Speedup | Query saving |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 7 | 0.40 | 16 | 0% | 26.92 | 2.45x | 39.22% |
| 5 | 5 | 0.55 | 16 | 0% | 27.92 | 2.36x | 39.84% |
| 5 | 7 | 0.55 | 16 | 0% | 36.25 | 1.82x | 44.20% |
| 5 | 5 | 0.70 | 16 | 0% | 37.50 | 1.76x | 44.46% |
| 5 | 7 | 0.70 | 8 | 0% | 52.83 | 1.25x | 50.31% |

A faster candidate (`gap=5`, `window=5`, `threshold=0.40`, `escalation=16`)
reactivated in 24.0 ticks on average for a 2.75x speedup, but produced one false
trigger in 12 stable runs (8.33%). It is intentionally **not** the recommended
reference.

More aggressive gap-3 policies reached still shorter delays but incurred much
higher observed false-trigger rates in parts of the grid, confirming that the
attention controller has a genuine Pareto surface rather than a free optimum.

## Interpretation

The v0.3-v0.6 line now closes a bounded self-organizing distinction cycle:

```text
learn questions' responses
        -> keep only a sufficient active basis
        -> sample the active basis densely
        -> revisit latent questions sparsely
        -> detect latent response drift
        -> temporarily escalate observation
        -> reactivate a question only through ordinary learned Delta + economy
```

This is a stronger basis for proceeding to learned connectivity between
surviving distinction neurons. The attention policy is still a finite research
reference; the next topology work should preserve this separation between
observation priority and representational authority.
