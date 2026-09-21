# Articulation Controller v0.1

Status: **experimental research controller**.

## Purpose

The articulation controller governs the boundary between Cortex's fuzzy,
resolution-dependent reasoning and exact local execution.

It does not answer semantic questions itself.

Its task is:

```text
current local articulation
        +
learned relational evidence
        +
estimated execution structure
        |
        v
 EXPLORE / EXECUTE / UNRESOLVED
                  |
                  +-- if EXECUTE:
                      native Cortex or LPU
```

## Epistemic states

### EXPLORE

Use when the representation is not stable enough to formalize.

Typical causes:

- source/entity support is weak;
- source articulation entropy is high;
- too few solved composition examples exist;
- no composition evidence exists yet.

`EXPLORE` means additional evidence or finer articulation may be useful. It is
not a negative assertion.

### UNRESOLVED

Use when Cortex has enough evidence to identify a genuine incompatibility, or
when no consistent exact path survives at the allowed evidence resolution.

Examples:

- substantial contradictory composition evidence;
- insufficient margin over a competing relation;
- a previously usable relational path has fallen below the minimum resolution.

`UNRESOLVED` prevents an exact processor from turning conflict into certainty.

### EXECUTE

Use only when the source articulation and relation composition pass the
epistemic gates and at least one path survives at an admissible resolution.

The controller then chooses an execution backend.

## Evidence gates

v0.1 uses transparent policy thresholds rather than a learned router.

Default gates include:

```text
minimum solved examples       4
minimum rule confidence       0.75
minimum competitor margin     0.25
minimum source support        0.70
maximum source entropy        0.35
minimum relation resolution   0.50
```

These are research policy parameters, not universal Cortex constants.

The composition learner is run permissively when feeding the controller. This
preserves competing hypotheses such as 60/40 evidence so the controller itself
owns the decision to execute or abstain.

## Resolution selection

The controller tests a descending resolution ladder and selects the highest
level at which the admitted relation composition remains executable.

This makes execution explicitly conditional on articulation resolution.

For example, if the weakest required evidence has support about 0.667, the
default ladder selects:

```text
epsilon = 0.65
```

rather than lowering immediately to the minimum 0.50.

Weak evidence is not deleted when it falls outside the selected articulation.

## Execution-cost profile

Before choosing a backend, the controller profiles the relation signature as a
layered directed acyclic graph.

It computes, without enumerating complete paths:

- path multiplicity;
- unique layered states;
- materialized transitions;
- reachable target basins.

The current heuristic prefers the LPU when:

```text
path multiplicity >= 32
and
path multiplicity / materialized edges >= 2
```

or when an exact proof/witness is explicitly required.

Otherwise native Cortex execution is used.

These thresholds are characterization parameters and should be tuned or learned
only after broader benchmarks.

## Crucial distinction: multiplicity is not ambiguity

Several reachable targets may all be valid.

The controller therefore does **not** treat multi-valued relations as epistemic
ambiguity.

Ambiguity concerns unstable identity or competing explanatory relation
structures, not the number of legitimate outputs.

## Authority boundary

The controller preserves the ordering:

```text
evidence
  -> Cortex basin articulation
  -> Cortex relational confidence
  -> controller decision
  -> exact execution
```

The LPU cannot:

- lower Cortex's evidence threshold;
- select a semantic relation;
- turn `EXPLORE` or `UNRESOLVED` into `EXECUTE`;
- increase premise confidence;
- write the exact result into persistent memory automatically.

An LPU witness is exact only conditional on the articulation supplied to it.

## Sigma-Lambda-inspired interpretation

The controller operationalizes a resolution boundary without making
Sigma-Lambda part of the Cortex formal specification.

A fuzzy informational state remains in exploration until a sufficiently stable
articulation exists. Exact logic is applied only inside that temporary
articulated structure. Returning to a finer or revised informational state is
always allowed.


## Validation result

The native-runtime campaign used 10 randomized seeds, width 8, and clean
composition depths 2 through 5.

All preregistered gates passed.

### Clean execution

Across 40 clean cases:

- execution rate: **1.000**;
- answer accuracy: **1.000**;
- backend-policy match: **1.000**;
- persistent-memory isolation: **1.000**;
- binding conflicts: **0**.

The controller selected:

```text
depth 2: Cortex   8 paths / 17 materialized edges
depth 3: Cortex  64 paths / 83 materialized edges
depth 4: LPU    512 paths / 147 materialized edges
depth 5: LPU   4096 paths / 211 materialized edges
```

For the structured knowledge evidence in this campaign, the highest viable
resolution was consistently:

```text
epsilon = 0.65
```

rather than the policy floor of 0.50.

### Epistemic controls

With 20% contradictory solved examples:

- controller state: **EXECUTE** in 100% of seeds;
- mean target-rule confidence: **0.8**.

With 40% contradictory examples:

- controller state: **UNRESOLVED** in 100% of seeds;
- reason: `conflicting-composition-evidence`.

With an intentionally ambiguous source articulation:

- controller state: **EXPLORE** in 100% of seeds.

### Exactness request

On otherwise inexpensive depth-2 cases, requesting an exact witness routed to
the LPU in **100%** of seeds and produced a valid witness.

This confirms that LPU selection is not purely a performance heuristic:
explicit exact-proof requirements can request it even when native Cortex
execution would otherwise be sufficient.

## Interpretation

v0.1 supports a three-state epistemic controller with a separate execution
backend decision:

```text
EXPLORE
    representation/evidence not stable enough yet

EXECUTE
    articulation stable
    -> Cortex when local exact traversal is cheap
    -> LPU when layered execution is advantageous or a witness is required

UNRESOLVED
    substantial incompatible evidence or no admissible exact articulation
```

This gives Cortex an explicit boundary between fuzzy cognition and exact local
computation without requiring semantic predicates or a neural orchestration
model.
