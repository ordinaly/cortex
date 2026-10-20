# Higher-Order Irreducibility v0.9

## Goal

Extend the learned distinction graph into a genuine 2-dimensional simplicial complex without equating a three-edge clique with a filled triangle.

The central invariant is:

> a 2-simplex is admitted only when synchronized three-probe responses contain a distinction that is invisible on every pairwise face.

This milestone is research-only. It does not claim a canonical information-theoretic definition of synergy.

## Operational criterion

For a probe triple `sigma = {i,j,k}` and contexts `x,y`, let `D_sigma(x,y)` be total-variation distance between their synchronized full triple-response distributions. For each pairwise face `f` of the triple, let `D_f(x,y)` be the analogous marginal TV distance.

A context pair can witness higher-order structure only when

`max_f D_f(x,y) <= face_tolerance`.

For such a pair, the residual is

`R_sigma(x,y) = max(0, D_sigma(x,y) - max_f D_f(x,y))`.

The simplex score is the maximum residual over supported context pairs.

This is a deliberately sufficient rather than necessary criterion. Differences already visible on a pairwise face are not counted as three-way irreducibility.

## Simplicial closure

A triangle is considered only if all three vertices are active and all three pairwise edges are already present in the v0.7/v0.8 distinction graph. Losing any face removes the 2-simplex immediately. Hysteresis applies only while the full boundary remains present.

Reference policy:

- `face_tolerance = 0.12`
- `simplex_on = 0.25`
- `simplex_off = 0.15`

These are controlled-fixture research values, not production defaults.

## Benchmark construction

Three persistent probes act on six contexts.

Four base contexts are deterministic one-bit perturbations of `000`:

- `000`
- `100`
- `010`
- `001`

This ensures that all three probes remain necessary under Probe Economy and that each pair of probes has mutual non-redundant distinction structure, producing a complete K3 boundary.

Two additional contexts test higher-order composition.

### Positive control

The final two contexts generate opposite noisy parity distributions. Their one- and two-probe marginals are identical, while their full triple distributions differ strongly.

### Negative control

The final two contexts both generate the same independent uniform three-bit distribution. The pairwise graph remains K3, but no genuine three-way distinction is present.

A naive clique-completion rule would fill the triangle in both controls. The v0.9 rule must fill it only in the positive control.

## Transition test

The same K3 boundary is held fixed while synchronized evidence changes from parity-synergistic to reducible and back again. The expected topology is:

`filled triangle -> triangle boundary only -> filled triangle`

without changing the three vertices or three pairwise edges.

## Acceptance criteria

Across development and held-out seeds:

- all runs retain all three probes;
- all runs retain the complete three-edge boundary;
- positive-control simplex activation rate is 100%;
- negative-control false-simplex rate is at most 5%;
- held-out positive mean residual is at least 0.50;
- held-out negative maximum residual remains below the 0.25 activation threshold;
- the fixed-boundary transition test removes and restores only the 2-simplex.
