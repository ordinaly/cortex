# Cortex Reasoning Map v1 — Planning / Replanning v1

Status: **frozen protocol / official campaign passed**.

Protocol ID: planning-replanning-v1.

## Research question

Can Cortex build a minimum-cost multi-step plan and then, after a local change
to the action model, repair only the affected region while preserving valid
prefix/suffix structure?

The intended transition is:

$$
\text{proof dependency}
\longrightarrow
\text{action dependency}
\longrightarrow
\text{local plan repair}.
$$

The central Cortex-specific hypothesis is not merely that a new plan can be
found. It is:

> A local model change should cause a local plan repair when an optimal local
> repair exists, rather than forcing a full restart from the initial state.

## Action model

A deterministic action is:

$$
a=(P_a,A_a,D_a,c_a),
$$

where:

- $P_a$ is the set of positive preconditions;
- $A_a$ is the add-effect set;
- $D_a$ is the delete-effect set;
- $c_a>0$ is an integer action cost.

For state $s$, action $a$ is applicable when:

$$
P_a\subseteq s.
$$

The successor is:

$$
T(s,a)=(s\setminus D_a)\cup A_a.
$$

A goal is a set of required facts $G$ and is satisfied when:

$$
G\subseteq s.
$$

## Base planner

The reference planner uses exact uniform-cost search with deterministic
tie-breaking.

The optimization criterion is total action cost, with step count used only as a
secondary tie-break.

This benchmark does not claim a novel search algorithm.

Uniform-cost search is the algorithmic oracle/reference used to isolate
replanning behavior.

## Temporal support provenance

Every valid plan receives an explicit support record.

For each action precondition, Cortex records the currently active producer:

- the initial state; or
- the most recent prior action that added the fact without a later deletion.

Final goal facts are annotated the same way.

This is the temporal planning analogue of explicit-proof-dependency-v1.

The support representation is not a monotone logical proof DAG because planning
contains delete effects. It is instead a temporally ordered dependency graph.

## Local repair rule

Let the original plan be:

$$
\pi=(a_0,\ldots,a_{n-1})
$$

with trajectory:

$$
s_0,s_1,\ldots,s_n.
$$

After an action-model change, Cortex preserves the longest prefix for which:

1. the current state is still exactly the expected trajectory state;
2. the action definition is unchanged;
3. the action remains applicable;
4. the resulting state is unchanged.

Let the first affected action index be $k$.

Cortex then searches from state $s_k$ for the **earliest reachable rejoin
state**:

$$
s_j,\qquad j>k,
$$

such that the old suffix beginning at $s_j$ is still unchanged and valid.

If such a rejoin exists, the repaired plan is:

$$
\pi'
=
(a_0,\ldots,a_{k-1})
\Vert
\rho_{k\rightarrow j}
\Vert
(a_j,\ldots,a_{n-1}),
$$

where $\rho_{k\rightarrow j}$ is a minimum-cost repair segment.

This prioritizes structural locality by rejoining the old trajectory as early
as possible.

## Global fallback

A local repair is not always possible.

If no old trajectory state can be rejoined while preserving a valid suffix,
Cortex performs a full uniform-cost replan from the updated initial state.

If that also fails, the result is **infeasible**.

Thus the epistemic outcomes are:

- unchanged;
- local-repair;
- full-replan;
- infeasible.

## Benchmark families

Seven structural families are frozen.

### action-failure

One middle action from the original plan disappears.

A two-action local detour reaches the immediately following old trajectory
state at equal cost.

Required behavior:

- preserve the unaffected prefix;
- use the detour;
- rejoin immediately;
- preserve the old suffix;
- match full-replan optimal cost.

### cost-shift

One middle action remains valid but becomes more expensive.

A previously suboptimal local detour becomes cheaper.

Required behavior:

- detect the action-definition change;
- replace the expensive action locally;
- improve cost relative to replaying the old action sequence;
- match the full-replan optimum.

### transition-rule

A middle action gains one unavailable precondition.

A local detour remains valid.

Required behavior:

- detect that the transition rule changed;
- avoid the changed action;
- repair locally;
- preserve the valid suffix.

### reversible-detour

The direct middle action disappears.

The replacement requires a reversible side trip:

1. leave the main trajectory;
2. temporarily lose a marker fact;
3. return with a temporary pass;
4. consume the pass while restoring the exact old trajectory state.

Required behavior:

- execute the side trip;
- restore the marker;
- consume the temporary pass;
- rejoin the old trajectory exactly.

### irrelevant-change

Only a dead-end distractor action changes cost.

The active plan and all of its support provenance remain unchanged.

Required behavior:

- return unchanged;
- preserve the exact plan fingerprint;
- preserve the exact support fingerprint.

### global-reroute

A middle action disappears, and from the preserved prefix state no old
trajectory state is reachable.

A valid alternate route exists, but it branches from an earlier state.

Required behavior:

- reject local repair;
- fall back to full replanning;
- find the alternate global route;
- match the full-replan optimum.

### infeasible

A required middle action disappears and no alternate route exists.

Required behavior:

- return infeasible;
- agree with a full search that no plan exists.

## Dead-end distractors

Every world contains several cheap dead-end branches reachable only near the
initial state.

They serve two purposes:

1. verify that minimum-cost planning avoids dead ends;
2. make local repair versus full restart measurable in state expansions.

The local repair search starts after the preserved prefix and therefore does not
re-explore unrelated early dead ends.

## Opaque naming

For every seed:

- state atoms are randomized opaque tokens;
- action IDs are randomized opaque tokens.

The planner receives only transition structure, costs, current state, and goal.

No family name is exposed through action or atom labels.

## Baseline

The baseline is exact full replanning from the updated initial state using the
same uniform-cost search implementation.

This is deliberately a strong baseline.

The benchmark does not ask local repair to find a cheaper plan than the global
optimum.

It asks local repair to:

- match the global optimum on the declared local-change fixtures;
- preserve unaffected plan structure;
- expand fewer states when local repair is sufficient.

## Development and official seeds

Development seeds:

$$
0,\ldots,9.
$$

Untouched official seeds:

$$
1400,\ldots,1419.
$$

With seven families, the official campaign contains:

$$
20\times7=140
$$

cases.

The official range must not run until the development campaign and frozen gates
pass unchanged.

## Frozen gates

The official campaign passes only if all of the following hold.

### Base planning

1. base-plan validity + support validity = **1.000**;
2. base planner selects the intended minimum-cost main chain = **1.000**;
3. dead-end action use = **0**.

### Correct replanning semantics

4. repair mode accuracy = **1.000**;
5. first-impact localization accuracy = **1.000**;
6. repaired/full-search feasibility agreement = **1.000**;
7. repaired plan cost equals full-replan optimum whenever feasible = **1.000**;
8. repaired-plan validity + support validity = **1.000**.

### Local repair

On action-failure, cost-shift, transition-rule, and reversible-detour:

9. local-repair rate = **1.000**;
10. rejoin occurs at the immediately following old trajectory state = **1.000**;
11. exact unaffected-prefix preservation = **1.000**;
12. at least one old suffix action is preserved = **1.000**;
13. required local replacement actions are used = **1.000**;
14. local search expands fewer states than full replanning in every case;
15. mean local/full expansion ratio <= **0.50**.

### Specialized controls

16. cost-shift improvement rate = **1.000**;
17. reversible state-restoration rate = **1.000**;
18. irrelevant-change exact plan/support stability = **1.000**;
19. global-reroute fallback correctness = **1.000**;
20. infeasible detection rate = **1.000**.

No gate may be weakened after official results are observed.

## Negative-result policy

If development reveals an implementation/protocol mismatch, it may be fixed
before any official run while preserving the scientific requirements above.

If the untouched official campaign fails, the result is retained. Any revised
official claim must use a new protocol version and fresh untouched seeds.

## Claim boundary

A passing result would support:

> Within declared finite deterministic action worlds, Cortex can compute
> minimum-cost plans, track temporal support dependencies, localize the first
> affected step after a model change, repair an affected plan segment while
> preserving valid prefix/suffix structure when a local optimum exists, fall
> back to global replanning when local repair is impossible, and report
> infeasibility when no plan exists.

It would not establish:

- planning in continuous spaces;
- probabilistic or partially observable planning;
- learned action-model discovery;
- optimal planning complexity improvements in general;
- large-scale automated planning superiority;
- human-like long-horizon planning.

## Planned files

- research/planning_replanning.py
- research/planning_replanning_campaign.py
- research/planning_replanning_gate.py
- tests/test_planning_replanning.py

The official campaign must remain untouched until development preflight is
green.


## Official outcome

The untouched official campaign on seeds **1400–1419** passed every frozen
gate across **140** cases.

On the **80** local-repair cases:

- mean local search expansions: **2.425**;
- mean full-replan expansions: **27.5**;
- mean local/full expansion ratio: **0.08745**;
- optimal-cost agreement: **1.000**;
- exact prefix and suffix preservation gates: **1.000**.

All specialized controls also passed, including cost adaptation, reversible
state restoration, irrelevant-change stability, global fallback, and
infeasibility detection.

See [PLANNING_REPLANNING_V1.md](PLANNING_REPLANNING_V1.md) and
[benchmarks/results/planning_replanning_v1_summary.json](../benchmarks/results/planning_replanning_v1_summary.json).

The frozen gates above were not changed after official evaluation.
