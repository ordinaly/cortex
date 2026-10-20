# Simplicial Integration v1.0 — Results

Status: **PASS**

The experiment reuses the v0.9 six-context world and compares two integrations over the same three active probes and same complete K3 boundary:

- boundary-only: vertex and edge sections only;
- full simplicial: vertex, edge, and learned 2-simplex sections.

## Holdout seeds 12–23

### Irreducible parity control

- complete K3 boundary: **12/12**;
- learned 2-simplex: **12/12**;
- mean boundary-only TV between contexts 4 and 5: **0.08794**;
- maximum boundary-only TV: **0.13672**;
- mean full simplicial TV: **0.80404**;
- minimum full simplicial TV: **0.77734**;
- mean simplex contribution: **0.71610 TV**;
- minimum simplex contribution: **0.66016 TV**;
- mean entropy reduction after applying the triple section: **0.37445 nats**;
- maximum local-to-global residual: **9.81e-10**.

The boundary model therefore remains nearly blind to the parity distinction, while the admitted 2-simplex changes the integrated articulation by a large and stable margin.

### Reducible control

- complete K3 boundary: **12/12**;
- learned 2-simplex: **0/12**;
- mean boundary-only TV: **0.08428**;
- mean full TV: **0.08428**;
- mean simplex contribution: **0.0**;
- mean entropy reduction: **0.0**;
- maximum local-to-global residual: **4.43e-10**.

Thus the integration layer does not manufacture higher-order structure merely because the graph contains a three-edge clique.

## Fixed-boundary transition

The 1-skeleton remained at exactly three edges throughout:

`filled -> boundary-only -> filled`

Integrated TV between contexts 4 and 5 changed as follows:

- initial filled regime: **0.796875**;
- reducible boundary-only regime: **0.117274**;
- restored filled regime: **0.828125**.

So the learned topology now has a direct computational consequence for `I`: a dimension-2 topological transition changes the global articulation while the vertices and pairwise graph remain fixed.

## Interpretation

This closes the immediate v0.9 gap. Higher-order topology is no longer diagnostic metadata; it participates in Cortex composition.

The current implementation remains a finite categorical maximum-entropy/IPF reference. It should be generalized only after its interaction with temporal transport has been tested.
