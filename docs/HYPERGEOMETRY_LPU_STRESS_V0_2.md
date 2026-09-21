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


## Results

The CI campaign used 10 seeds, depths 2 through 5, and branching width 8.
All preregistered gates passed.

### Semantic equivalence

Across 40 clean cases:

- Cortex enumerating executor accuracy: **1.000**;
- Cortex + LPU accuracy: **1.000**;
- executor agreement: **1.000**;
- exact LPU witness validity: **1.000**;
- Cortex memory unchanged by LPU execution: **1.000**.

Raw relation-agnostic LPU reachability was ambiguous in **100%** of cases:
the intended target and all three decoys were reachable. Exact reachability
alone therefore did not recover the semantic answer.

### Contradiction and uncertainty

With 20% contradictory solved examples, the learned target composition retained
mean confidence **0.8** and held-out accuracy remained **1.000**.

With 40% contradictory examples, the target composition fell below the
acceptance threshold in **100%** of cases. Cortex returned unresolved and the
LPU received no semantic execution request.

This is the intended authority boundary: the LPU cannot turn uncertain Cortex
structure into a fact.

### Branching depth

At depth 5 and width 8, the relation-conditioned lattice contains:

```text
4096 complete matching paths
211 materialized layered edges
```

The enumerating Cortex reference executor must represent the 4096 completed
paths. The Cortex + LPU formulation collapses equivalent intermediate states
before exact reachability.

Hosted-runner timing characterization at depth 5:

```text
enumerating executor median: 24.758 ms
Cortex + LPU median:          0.676 ms
ratio:                       36.95x
```

These timings are environment-specific characterization, not a universal
performance claim. The stronger algorithmic observation is the reduction in
represented path multiplicity.

### Plasticity controls

- multi-valued relation result preserved: **passed**;
- lowering one required relation below the articulation resolution removed the
  inference before LPU invocation: **passed**;
- zero binding conflicts across all native-runtime worlds: **passed**.

## Architectural conclusion

The experiment supports the following separation:

```text
Cortex:
    discover basins
    learn relational hypergeometry
    estimate confidence
    choose resolution
    articulate a relation signature

LPU:
    receive a temporary articulated graph
    perform exact local computation
    return a deterministic witness
    forget the temporary graph
```

The LPU is therefore useful as an exact coprocessor **after articulation**, not
as an ontology, knowledge store, or independent reasoner.
