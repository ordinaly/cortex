# Distinction Connectivity v0.7 — Results

## Outcome

**Focused pass.** The first learner-visible pairwise topology rule behaves as
intended on both synthetic regime-switch tests and the noisy delayed benchmark.

The result is stronger than merely obtaining a positive edge: in the real
14-probe fixture, the two probes selected by Probe Economy each contribute two
context-pair distinctions that the other does not.

## Noisy delayed H=3 benchmark

The benchmark reuses the established fixture with:

```text
14 declared probes
2 probes in the exact empirical minimum sufficient basis
6 required context-pair separation edges
```

Probe Economy again selected declared-probe indices:

```text
[9, 13]
```

Pairwise topology measurements:

| Measure | Result |
|---|---:|
| Active probes | 2 |
| Required context-pair edges | 6 |
| Unique edges contributed by probe 9 | 2 |
| Unique edges contributed by probe 13 | 2 |
| Mutual complementarity Gamma | 1/3 |
| Joint coverage gain over best single probe | 1/3 |
| Learned probe-neuron edges | 1 |

The single graph edge is exactly `(9,13)`.

Since the default research edge-on threshold is 0.10, the observed score
`0.3333` has a substantial margin over the activation boundary.

## Same vertices, different topology

A separate controlled unit test keeps the same two distinction neurons alive
while changing one probe's empirical response.

Initially:

```text
q0 partitions a | bc
q1 partitions ab | c
```

Each probe has one unique separation relative to the other, so the graph forms
an edge.

After the regime change, q1 duplicates q0. The vertex set remains unchanged but
mutual complementarity falls to zero; the edge is retired.

This is direct implementation evidence for the architectural idea:

```text
same distinction-neuron substrate
        + different observed phenomenon
        -> different active topology
```

## Interpretation

v0.7 supports a limited claim:

> Cortex can infer a dynamic pairwise graph from the current nonredundant
> structure of learned distinctions, without connecting probes merely because
> they correlate or agree.

It does not yet establish an optimal topology-learning rule, nor does an edge
prove irreducible higher-order synergy. The score is a finite complementarity
reference based on merge-veto coverage.

The next experiment should stress edge stability under noisy drift and multiple
active probes before introducing triangles or higher simplices. Higher-order
cells should only be admitted when they contribute structure not already
accounted for by their faces.
