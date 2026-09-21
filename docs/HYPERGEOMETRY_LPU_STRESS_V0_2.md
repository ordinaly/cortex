# Hypergeometry + LPU Stress Test v0.2

Status: **experimental research campaign**.

## Question

The relational hypergeometry has already demonstrated held-out composition.
This campaign asks a different question:

> When does Cortex benefit from an exact Logical Processing Unit?

The hypothesis is deliberately narrow:

```text
Cortex discovers and weights structure.
Cortex chooses an articulated relation signature.
LPU executes that temporary exact subproblem.
LPU returns a witness.
```

The LPU does not discover relations, select semantic rules, assign confidence,
or write to Cortex memory.

## Exact layered execution

For a Cortex-learned relation signature

```text
R0 -> R1 -> R2 -> ... -> Rk
```

the controller constructs a temporary layered graph.

A state is:

```text
(depth, concept-basin)
```

and an edge from depth $i$ to $i+1$ is materialized only when Cortex's current
hypergeometry contains the relation/orientation required by step $i$ at the
chosen evidence resolution.

The LPU then evaluates exact `REACHABLE` between

```text
(0, source)
```

and

```text
(k, candidate)
```

and returns a deterministic path witness.

Thus exactness is conditional on Cortex's articulation. The LPU cannot elevate
a low-confidence edge above the chosen resolution.

## Why this can matter computationally

The current Cortex research executor enumerates every matching path.

A layered graph can instead merge identical intermediate states. In a width
$w$, depth $d$ reconverging lattice, the number of matching paths is

```math
w^(d-1)
```

while the number of distinct layered states is only linear in $d w$, and the
materialized graph remains polynomial.

The campaign therefore records both runtime characterization and structural
counts. Runtime on hosted CI is not treated as a hard performance claim.

## Controls

The campaign includes:

- Cortex enumerating executor;
- Cortex + LPU layered exact executor;
- relation-agnostic LPU reachability;
- relation-name randomization;
- 20% contradictory composition examples;
- 40% contradictory examples;
- multi-valued targets;
- resolution-driven revision;
- memory-isolation checks;
- depths 2 through 5 with reconverging branching.

Relation-agnostic LPU is intentionally given raw graph reachability where every
candidate is reachable through some path. If it cannot distinguish the
relation-conditioned target, that demonstrates why exact computation is not a
world model.

## Expected interpretation

A useful LPU outcome is **not necessarily higher accuracy** than the exact
Cortex reference executor. On clean tasks both should agree.

Evidence for a useful boundary is instead:

- identical semantic answer under the same Cortex articulation;
- exact deterministic witnesses;
- zero persistent-memory mutation;
- graceful refusal when Cortex does not resolve a composition;
- preservation of multi-valued relations;
- fewer materialized states/edges than fully enumerated path multiplicity at
  larger depth.

This keeps epistemic authority in Cortex and computational exactness in the LPU.
