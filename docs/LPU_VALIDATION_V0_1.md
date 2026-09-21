# LPU Validation v0.1

Status: **active validation campaign**.

## Purpose

This campaign validates the Logical Processing Unit before Cortex's autonomous
learning campaign begins.

The LPU is not evaluated as an independent reasoner. The target contract is
narrower:

> given a temporary articulated problem, return an exact local result and
> witness without creating or mutating Cortex world knowledge.

## Hard gates

The campaign fails if any of the following occur:

- Boolean truth-table mismatch;
- exact rational arithmetic/comparison mismatch;
- graph reachability disagreement with an independent oracle;
- invalid positive witness or non-empty negative witness;
- result changes when the same graph edges are supplied in another order;
- result changes after unrelated LPU calls;
- any Cortex runtime snapshot mutation caused by LPU calls;
- failure to preserve two simultaneously supplied values of the same
  relation-like structure.

Absolute runtime is reported only as characterization because hosted CI timing
is not stable enough for a scientific performance threshold.

## Campaigns

### Exact logic and arithmetic

All Boolean inputs are exhaustively tested. Thousands of random rational
operations are compared to Python's independent `Fraction` implementation.

### Graph oracle and witness validation

Random directed graphs at multiple sizes are queried for reachability. Results
are checked against an independent BFS oracle. Every positive result must carry
a path whose edges all exist in the supplied graph.

The same query is repeated after random permutations of edge input order.

### Purity and Cortex isolation

A native `CortexRuntime` is given observations and snapshotted. Thousands of
LPU calls are executed. The Cortex snapshot before and after must be byte-for-byte
identical.

This is the key architectural invariant:

```math
L(S_t,q) \not\Rightarrow \Delta M_{Cortex}.
```

### Ontology-plasticity control

The temporary structure contains both

```text
Country-A -> capital -> X
Country-A -> capital -> Y
```

represented as two supplied relation-like targets. Both must remain valid
inputs to exact computation. The LPU contains no rule that a relation named
"capital" is single-valued.

This test does not prove open-world learning; it verifies that the exact unit
does not impose that ontology constraint.

## Performance characterization

Reachability latency is measured at 8, 32, 128 and 512 nodes. These values are
recorded for later regression comparisons but are not pass/fail gates in v0.1.

## Promotion criterion

LPU v0.1 is considered safe to carry into the autonomy campaign only if every
semantic/isolation gate passes. Automatic LPU invocation remains out of scope:
Cortex learning and orchestration should be tested separately after the
coprocessor boundary is validated.
