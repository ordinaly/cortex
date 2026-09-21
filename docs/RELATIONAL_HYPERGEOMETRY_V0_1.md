# Relational Hypergeometry v0.1

Status: **experimental research geometry**.

## Purpose

Cortex already represents fuzzy entity attractors and sparse relation evidence.
This experiment makes their intended geometry explicit without replacing
Cortex's learned state with a symbolic ontology.

The design is inspired by Sigma-Lambda principles, but it remains a distinct
Cortex research mechanism. In particular:

- articulation is resolution-dependent;
- uncertainty remains soft rather than collapsing to a hard predicate;
- weak structure can disappear from the current articulation without being
  deleted from memory;
- a concept may be both an attractor and a participant in higher-order
  relations;
- multi-valued and changing relations remain admissible;
- local structure is preferred over fixed global semantic dimensions.

## Concept basins

Every world symbol admitted by Cortex remains an ordinary concept/entity basin.
This includes values normally called relations:

```text
France
Paris
capital
continent
Europe
...
```

The word `capital` is therefore not merely an edge label. It is a concept
basin with its own identity and evidence neighborhood.

## Higher-order relational edge

A statement is interpreted geometrically as one soft hyperedge:

```text
             capital
              /   \
             /     \
        France --- Paris
```

Formally:

```math
e = (s, r, o, w)
```

where subject, relation and object are all basins and $w$ is the support of the
complete stored statement motif.

For the current reified knowledge grammar, $w$ is the minimum posterior support
among the structural edges required to recover the statement. This is a
conservative bottleneck composition: a statement is no stronger than its
weakest structural dependency.

## Incidence identity versus belief

A genuine hypergraph needs persistent incidence identity. Cortex therefore
records the structural identity of each knowledge statement token:

```text
statement-id -> (subject basin, relation basin, object basin)
```

This incidence record does **not** assert that the statement is true. Its
current geometric support is recomputed from Cortex's live evidence over the
motif dependencies.

Consequently, evidence revision changes the hyperedge weight while preserving
the identity of the relation instance:

```text
same incidence
    support 0.67 -> 0.20
```

This separation prevents the geometry from reconstructing statement identity
from arbitrary weak pairwise co-observation edges, and it preserves revision:
the topology can weaken, disappear at a chosen resolution, and later return
without inventing a new semantic object.

## Relation basin as local direction field

The relation basin also induces role-aware local directions.

For observed statements

```text
France  -- capital --> Paris
Germany -- capital --> Berlin
Japan   -- capital --> Tokyo
```

the `capital` basin contains examples of a recurring local transport field.

The relation is **not** assumed to be a mathematical function. If Cortex has

```text
Canada -- official language --> English
Canada -- official language --> French
```

then both target directions coexist with soft support.

## Resolution topology

Let a hyperpath $pi$ connect two concept basins. Its support is the weakest
hyperedge on that path:

```math
kappa(pi) = min_(e in pi) w(e).
```

The strongest available connection is:

```math
kappa(a,b) = max_(pi:a to b) kappa(pi).
```

Cortex defines the relational resolution distance

```math
d_R(a,b) = 1 - kappa(a,b).
```

At articulation resolution $epsilon$, only hyperedges with support at least
$epsilon$ participate in the visible H0 component.

Changing $epsilon$ never mutates or deletes the underlying hyperedges. A weak
relation can therefore disappear from a coarse articulation and reappear when
the resolution is relaxed.

The max-min connectivity satisfies the ultrametric-style inequality

```math
d_R(a,c) <= max(d_R(a,b), d_R(b,c)).
```

This is checked directly by the validation campaign.

## Two-level geometry

Cortex now exposes two complementary structures:

```text
base basin geometry
    identity / prototype / reliability / noise

soft relational hypergeometry
    higher-order incidence / relation directions / resolution connectivity
```

The random identity vectors used by the first knowledge adapter only separate
basins. They do **not** determine semantic proximity. Semantic neighborhood
comes from learned relational incidence.

This distinction is important: `capital` and `Paris` need not become close
in the raw sensory metric. They are adjacent in relational geometry because
they participate in a supported higher-order configuration.

## Local articulation

A knowledge local articulation now includes:

- fuzzy entity support from the base Cortex articulation;
- local feature reliability/noise;
- native relation and causal evidence;
- decoded property/value coordinates;
- the incident relational hypergeometry at a chosen resolution.

Thus the intended read operation is:

```text
point x
  -> basin support
  -> local evidence
  -> soft incident hyperedges
  -> relation-conditioned directions
  -> resolution-dependent neighborhood
```

## Scope boundary

v0.1 is intentionally read-only. Relational evidence deforms the **interpreted
geometry** but does not yet feed back into native entity binding.

That separation is deliberate. Promotion into the Rust binding/update loop
should happen only after the geometry is shown to improve held-out inference,
revision behavior, and/or representation economy without reducing plasticity.

The next scientific experiment should compare:

```text
identity geometry only
vs
identity + relational hypergeometry
```

on held-out general-knowledge inference and changed-world revision.
