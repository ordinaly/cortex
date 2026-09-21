# Cortex Autonomy Campaign v1

Status: **active experimental campaign**.

## Central question

After a finite chronological learning stream ends and the teacher is removed,
can Cortex continue to infer, absorb novelty, revise, preserve unaffected
memory, and decide when exact reasoning is justified using only its own
persistent state?

The campaign is deliberately structured-data-first. Language-model quality is
not allowed to confound the cognitive test.

## Hard teacher-detachment boundary

The campaign has two eras.

### Childhood

Cortex receives solved structural examples and incomplete future query worlds.
This is the only phase allowed to contain solved examples specifically intended
to establish reusable relation compositions.

### Autonomous phase

After detachment:

- there are zero neural/model calls;
- held-out answer relations are never supplied;
- query labels are never fed back into memory;
- new environmental observations may still arrive;
- changed-world outcomes may contradict old regularities;
- all inference is produced from Cortex memory, relational hypergeometry,
  controller state, and optionally the stateless LPU.

Environmental feedback is not treated as a teacher model. For example, Cortex
may observe that a new world event occurred, but it is never shown the answer
to the held-out query that evaluates its inference.

## Campaign phases

### 1. Teacher-free held-out inference

Childhood contains six solved examples of an opaque two-step relation
composition and four incomplete query worlds.

After detachment Cortex must infer the missing result relations. Direct-memory
lookup is checked separately and must remain empty.

### 2. Post-detachment novelty

Entirely new entity basins are introduced after teacher removal. Cortex receives
only component relations and must reuse childhood structure to infer held-out
consequences.

### 3. Legitimate multi-value world change

One previously single-valued relation acquires a second legitimate target.

The desired behavior is not conflict or replacement. Cortex should preserve
both outcomes and infer both higher-order consequences.

This is the explicit control for cases such as a country legitimately acquiring
a second capital.

### 4. Deep autonomous composition

Childhood establishes an opaque depth-four composition. After detachment a new
width-four reconverging lattice is observed with no direct answer relation.

The query has 64 valid relation-conditioned paths. This tests both deep transfer
and whether the calibrated controller chooses the LPU when the exact subproblem
benefits from layered compression.

### 5. Changed-world conflict

New independent outcomes gradually contradict the childhood composition while
the held-out query itself remains unlabeled.

The intended calibrated-controller trajectory is:

    EXECUTE -> EXPLORE -> UNRESOLVED

At moderate contradiction, the learned v0.2 controller should keep the issue
open. At strong conflict it should recognize genuine incompatibility.

The no-controller ablation is expected to keep executing through the conflict,
which is precisely the risk the controller is intended to prevent.

### 6. Recovery

Contradictory observations are not deleted. The environment later supplies
enough independent supporting outcomes for the original composition to become
strongly dominant again.

The target trajectory is:

    EXECUTE -> EXPLORE -> UNRESOLVED -> EXECUTE

This is a direct plasticity test: exact articulation must be reversible in both
directions.

### 7. Unaffected-memory preservation

A disjoint relation family is learned during childhood. Its held-out query is
checked before, during, and after the changed-world conflict.

Revision should remain local rather than destabilizing unrelated memory.

### 8. Memory-pressure characterization

Separate runtimes sweep maximum entity capacity at 256, 512, and 1024 entities
while unrelated observations accumulate.

The 1024-entity case is a gate for retained core inference. Lower capacities are
reported as characterization rather than forced to pass.

## Ablations

Every main query is evaluated under the same Cortex memory with:

- no articulation controller, LPU unavailable;
- no articulation controller, LPU available;
- fixed controller v0.1, LPU unavailable;
- fixed controller v0.1, LPU available;
- calibrated controller v0.2, LPU unavailable;
- calibrated controller v0.2, LPU available.

LPU availability does not grant semantic authority. Under controller modes, the
LPU is called only when the controller selects it. Disabling the LPU forces the
same admitted articulation through the Cortex reference executor.

## Primary gates

The first campaign requires:

- zero teacher/model calls after detachment;
- zero held-out answer leakage;
- zero binding conflicts in the main stream;
- exact held-out inference after childhood;
- exact transfer to post-detachment novel entities;
- legitimate multi-value preservation;
- correct depth-four autonomous inference;
- LPU selection plus exact witness on the deep task;
- calibrated state trajectory
  EXECUTE -> EXPLORE -> UNRESOLVED -> EXECUTE;
- unaffected-memory preservation;
- semantic equivalence with LPU disabled;
- retained inference in the high-capacity memory-pressure condition.

## Claim boundary

Passing supports a narrow claim of teacher-free structural autonomy on the
declared structured fixtures.

It does not establish natural-language autonomy, unrestricted world modeling,
or autonomous goal formation. The structured identity adapter still supplies
opaque basin identity, and the current changed-world test concerns learned
relational compositions rather than arbitrary temporal semantics.
