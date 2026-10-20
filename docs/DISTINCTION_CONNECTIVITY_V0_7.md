# Distinction Connectivity v0.7

## Status

Research-only first topology-learning stage after learned Delta, probe economy,
sampling economy, and latent reactivation. It does not modify `spec-v0.9.5` or
the native runtime.

## Goal

Cortex now has a changing population of learned distinction neurons. v0.7 asks
when two surviving neurons should be connected.

The reference rule deliberately rejects correlation as sufficient evidence for
an edge. Two probes connect only when **both contribute current separation
structure that the other cannot provide**.

## Learner-visible separation sets

Probe Economy already defines, for each probe `q`, a set of context-pair merge
vetoes:

```text
S_q = {(x,y): q currently prevents x and y from being folded together}.
```

Let

```text
S = union_q S_q
```

be the full current empirical distinction relation.

For probes `i,j`, define their unique contributions relative to one another:

```text
U_i|j = S_i \ S_j
U_j|i = S_j \ S_i
```

and the v0.7 pairwise complementarity score

```text
Gamma(i,j) = min(|U_i|j|, |U_j|i|) / |S|.
```

Equivalently for two sets, this is the joint coverage gained beyond the better
single probe, normalized by the full required relation.

Properties of this finite reference score:

- `Gamma(i,j)=0` if either probe is redundant with or subsumed by the other;
- `Gamma(i,j)>0` only if each probe adds at least one distinction the other lacks;
- the score uses learner-visible current evidence only;
- the score can change while the probe identities remain fixed.

This is a conservative complementarity measure, not a claim of canonical
information-theoretic synergy.

## Dynamic graph

`HystereticDistinctionGraph` uses the currently active Probe Economy population
as vertices.

For each active pair:

```text
Gamma >= edge_on   -> add edge
Gamma <  edge_off  -> remove existing edge
```

with `edge_off <= edge_on` to avoid topology chatter near the threshold.

If a probe itself leaves the active representation, all incident edges disappear
immediately. Hysteresis protects uncertain relation evidence, not nonexistent
vertices.

The default research thresholds are:

```text
edge_on  = 0.10
edge_off = 0.05
```

They are reference values, not yet calibrated under noisy topology drift.

## Same neurons, different topology

A core v0.7 unit test keeps the same two probes active across a regime change.
Initially:

```text
q0 partitions a | bc
q1 partitions ab | c
```

so each contributes a unique merge-veto edge and the pair connects.

Then q1 changes until it duplicates q0. The vertices remain `(q0,q1)`, but
mutual complementarity falls to zero and the graph loses the edge.

This directly realizes the architecture hypothesis:

> one persistent distinction-neuron substrate can induce different active
> topology depending on the phenomenon currently being observed.

## Noisy delayed benchmark

The first non-toy benchmark reuses the noisy delayed horizon-3 world from the
learned-Delta and Probe Economy line.

That world has:

```text
14 declared probes
2 probes in the exact learner-visible minimum sufficient basis
```

Because two probes are required to cover the full learned distinction relation,
neither can subsume the other. v0.7 therefore requires:

- both selected probes contribute at least one unique context-pair distinction;
- positive mutual complementarity;
- exactly one graph edge between the two active probes;
- full Probe Economy coverage remains 1.0.

## Boundary

v0.7 learns pairwise edges only. It does not yet infer triangles or higher
simplices, and an edge does not mean the two neurons form an irreducible
higher-order computation. It means only that their current distinction coverage
is mutually complementary.

The next step should challenge pairwise topology under noise and phenomenon
switches, then define a higher-simplex criterion that requires gain not already
explained by all lower-dimensional faces.
