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
