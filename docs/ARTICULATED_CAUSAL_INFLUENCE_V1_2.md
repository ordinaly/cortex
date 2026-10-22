# Articulated Causal Influence v1.2

## Goal

Promote `C = causal influence` from scalar event outcomes to controlled changes in integrated Cortex articulations.

The operator should answer:

> How distinguishable are the post-condition articulations under a valid intervention versus its control?

Temporal succession alone is not causal evidence. v1.2 therefore accepts only explicitly marked intervention/control samples. A causal interpretation requires exogenous/randomized assignment or an otherwise valid controlled design.

## Effect geometry

Each observation contains a `SimplicialIntegrationRead` produced downstream of `I`.

For all probes shared across the retained intervention and control samples, Cortex computes empirical mixture marginals for each support up to the configured dimension. The reference policy weights:

- dimension 0: `0.15`;
- dimension 1: `0.25`;
- dimension 2: `0.60`.

The causal influence score is the weighted mean total-variation distance between intervention and control articulation mixtures.

This differs intentionally from v1.1 temporal transport:

- `K` protects lower-dimensional geometry so identity may survive rearticulation;
- `C` is allowed to inspect dimension 2 so an intervention can causally alter higher-order composition without destroying identity.

## Benchmark

Three cases are evaluated on development and held-out seeds.

### 1. Low-dimensional genuine effect

Control and intervention have identical independent higher-order structure, but the stable marker changes from probability `0.20` to `0.80`. `C` should detect a large effect.

### 2. Higher-order genuine effect with preserved identity

Control has EVEN parity and intervention has ODD parity while all singleton and pairwise parity marginals are unchanged and the marker remains at `0.50`.

Expected result:

- v1.1 `K` distance remains small because dimensions 0 and 1 are stable;
- v1.2 `C` detects the intervention in dimension 2.

This demonstrates that `same continuing entity` does not imply `causally unchanged`.

### 3. Confounded association / randomized null

A passive exposure proxy is constructed with a strong `0.20` versus `0.80` association. Separately, the randomized intervention/control groups both use marker probability `0.50`.

The observational association may be large, but the controlled causal influence must remain near zero.

## Acceptance

On held-out seeds:

- minimum genuine marker effect > `0.30`;
- minimum higher-order causal effect > `0.10`;
- maximum temporal identity distance for the parity intervention < `0.10`;
- mean dimension-2 parity effect > `0.20`;
- maximum dimension-0 and dimension-1 parity effects < `0.05`;
- maximum randomized-null causal effect < `0.05`;
- minimum passive association > `0.30`;
- passive association is at least 6x the mean randomized-null effect.

## Scope

This is a controlled experimental causal estimator. It does not recover interventions from passive streams, solve hidden confounding, or implement general do-calculus. Those require a later causal-discovery layer and should not be conflated with the validity of the intervention-sensitive articulation metric itself.
