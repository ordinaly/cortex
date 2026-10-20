# Simplicial Integration v1.0

## Goal

Make the learned distinction complex computationally meaningful for the fifth operator,
`I = integration/composition`.

The v0.9 complex distinguishes a three-edge boundary from a genuinely irreducible filled
2-simplex. v1.0 asks whether that coface changes the integrated articulation in a way its
boundary cannot.

## Local-to-global rule

For one context, each active simplex supplies an empirical marginal distribution over the
responses of its probe neurons:

- vertices supply one-probe marginals;
- edges supply pairwise marginals;
- active 2-simplices supply synchronized triple marginals.

Cortex reconstructs the least-committal global distribution consistent with those local
sections using iterative proportional fitting (IPF). Equivalently, among feasible global
categorical distributions it selects the maximum-entropy articulation.

A K3 boundary therefore does **not** imply a filled triangle. Without the coface, only the
three pairwise marginals constrain the global articulation.

## Benchmark contract

The benchmark reuses the v0.9 six-context world.

Four base contexts force all three probes to remain active and maintain a complete K3
boundary. Contexts 4 and 5 are then compared in two regimes:

1. **Irreducible control:** opposite noisy parity distributions. All one- and two-probe
   marginals are equal in the population limit, while the triple joint distribution differs.
2. **Reducible control:** identical independent uniform three-bit distributions.

Thus the vertex set and pairwise graph are the same. The only legitimate source of extra
integrated distinction in the positive control is the learned 2-simplex.

The principal comparison is

`D_boundary(4,5)` versus `D_filled(4,5)`,

where distance is total variation between the reconstructed global articulations.

## Acceptance

On held-out seeds:

- all K3 boundaries remain complete;
- positive controls retain the v0.9 simplex;
- negative controls remain boundary-only;
- positive boundary-only TV is below 0.20;
- positive filled TV is above 0.60;
- the simplex contributes at least 0.45 additional TV;
- negative controls gain no artificial dimension-2 distinction;
- IPF local-to-global residual remains below `1e-7`;
- during `filled -> boundary -> filled`, all three graph edges remain fixed while the
  integrated articulation loses and regains the higher-order distinction.

## Scope

This is a finite categorical reference implementation. It is not yet a general sheaf,
continuous-state, or arbitrary-dimensional integration engine. Its purpose is to establish
that learned topology can causally affect Cortex articulation before generalizing the
mechanism.
