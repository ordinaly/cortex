# Learned Distinguishability under Noise v0.2

Status: **research characterization / stochastic controlled worlds**.

This experiment keeps the learned predictive `Delta` mechanism from v0.1 unchanged and moves the evaluation from deterministic observations to noisy controlled worlds.

The learner still receives only opaque contexts, declared action probes and sampled output words. Hidden mixtures, transition kernels and exact predictive profiles remain evaluator-only.

## Question

Can empirical trace distinguishability refine at the correct observational horizon without either:

- hallucinating distinctions from sampling noise; or
- collapsing distinctions that become predictively accessible later?

## Exact refinement ladders

Two noisy controlled families are used.

### Immediate family

At horizon 1 all four evaluator contexts are predictively aliased:

```text
((0, 1, 2, 3),)
```

At horizon 2 action-conditioned temporal evidence separates exactly two classes:

```text
((0, 2), (1, 3))
```

### Delayed family

The hidden distinction is deliberately postponed.

Horizon 1:

```text
((0, 1, 2, 3),)
```

Horizon 2:

```text
((0, 1, 2, 3),)
```

Horizon 3:

```text
((0,), (1,), (2,), (3,))
```

Thus success requires both restraint and refinement: Cortex must preserve an alias while no accessible probe separates it, then split only after the longer probe family carries evidence.

## Sampling protocol

For each evaluator context and every declared probe, v0.2 samples 1,024 independent controlled trials using fixed development seeds. The learner uses:

- `epsilon = 0.25`;
- `min_samples = 128`;
- finite-budget confidence guards;
- complete-link quotient proposals.

These are development parameters, not universal constants.

## Acceptance

Each learned partition must:

1. equal the exact finite-oracle quotient for that horizon;
2. pass `audit_partition` at epsilon 0.25;
3. use the oracle-minimal feasible number of cells.

The five cases are evaluated jointly as one refinement ladder.

## Interpretation boundary

A successful v0.2 result shows that the current empirical distinguishability learner can recover these small stochastic predictive quotients under the stated sampling budget. It does not establish consistency for arbitrary stochastic processes or nonstationary streams.

The probe family remains declared rather than discovered.

## Next decision

If the noisy refinement ladder passes, the next useful question is **probe economy** rather than adding more raw capacity.

We should let candidate probe neurons compete by a cost-aware criterion such as:

```text
predictive distinction gain - complexity cost
```

and ask whether Cortex can retain a minimal useful distinguishability population while preserving the oracle quotient and held-out predictive performance.
