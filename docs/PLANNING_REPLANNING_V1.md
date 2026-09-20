# Cortex Reasoning Map v1 — Planning / Replanning v1

Status: **passed official benchmark / research-only**.

Protocol: planning-replanning-v1.

Official run: GitHub Actions run 35510844476, job 106078366741.

## Question

Can Cortex compute a minimum-cost multi-step plan and, after a local action-model
change, repair only the affected part while preserving unaffected plan
structure?

The benchmark evaluates four distinct outcomes:

- unchanged;
- local repair;
- full replanning fallback;
- infeasible.

## Official campaign

Untouched official seeds:

$$
1400,\ldots,1419.
$$

Seven world families produced:

$$
20\times7=140
$$

official cases.

The four local-change families produced **80 local-repair cases**.

## Result

Every frozen gate passed.

| metric | result |
|---|---:|
| base-plan/support validity | **1.000** |
| repair-mode accuracy | **1.000** |
| first-impact localization | **1.000** |
| repaired/full-search feasibility agreement | **1.000** |
| optimal-cost agreement | **1.000** |
| feasible repaired-plan/support validity | **1.000** |
| local-repair rate | **1.000** |
| immediate trajectory rejoin rate | **1.000** |
| exact prefix preservation | **1.000** |
| suffix preservation | **1.000** |
| local replacement correctness | **1.000** |
| local search-saving rate | **1.000** |
| cost-shift adaptation | **1.000** |
| reversible state restoration | **1.000** |
| irrelevant-change stability | **1.000** |
| global-fallback correctness | **1.000** |
| infeasible detection | **1.000** |

## Search work

On the 80 official cases where local repair was possible:

- mean local repair search: **2.425 expanded states**;
- mean full replanning search: **27.5 expanded states**;
- mean local/full expansion ratio: **0.08745**.

Equivalently, the local repair search expanded about **91.25% fewer states** on
these declared fixtures.

This is a state-expansion result, not a general runtime-complexity theorem.

## Local action failure

When one middle action disappeared, Cortex preserved the unaffected prefix,
found the two-step detour, rejoined the immediately following old trajectory
state, and preserved the remaining suffix.

The repaired plan matched full uniform-cost replanning in total cost on every
official seed.

## Cost revision

When one still-valid action became expensive, Cortex treated the action
definition change as an impact point rather than blindly replaying the old
sequence.

It replaced the expensive step with the now-cheaper detour in every official
case.

The repaired cost was strictly lower than replaying the old action sequence
under the new costs, and equal to the full-replan optimum.

## Transition-rule change

When a middle action gained a new unavailable precondition, Cortex localized
the first affected step and replaced only that segment.

The valid downstream suffix remained intact.

This is the planning analogue of proof-dependency invalidation:

$$
\text{changed dependency}
\Rightarrow
\text{local downstream repair}.
$$

## Reversible detour

The reversible family required leaving the main trajectory, temporarily deleting
a marker fact, returning with a temporary pass, and consuming that pass while
restoring the exact old trajectory state.

All official cases restored the persistent marker, removed the temporary pass,
rejoined immediately, and completed the original suffix.

This verifies that local repair is not restricted to monotone add-only action
chains.

## Irrelevant change

Changing the cost of an unused dead-end action left:

- the plan fingerprint unchanged;
- the temporal support fingerprint unchanged.

Irrelevant-change stability was **1.000**.

This is an important locality control: the planner does not rewrite an active
plan merely because some unrelated part of the action model changed.

## Global fallback

In the global-reroute family, the changed middle action made every old
trajectory state unreachable from the preserved prefix.

A valid alternate route still existed, but it branched from an earlier state.

Cortex correctly rejected local repair and invoked full replanning in every
case.

Global-fallback correctness was **1.000**.

## Infeasibility

In the infeasible family, no local detour and no global alternate route existed.

Both the repair system and the full uniform-cost baseline returned no plan.

Infeasible-detection rate was **1.000**.

## Temporal support provenance

Every feasible plan carries explicit provenance for:

- every action precondition;
- every final goal fact.

Each support points to either:

- the initial state; or
- the most recent prior action that produced the currently active fact.

Because planning contains delete effects, this is a temporally ordered support
graph rather than the monotone proof DAG used by explicit-proof-dependency-v1.

This provides the bridge:

$$
\text{proof dependency}
\rightarrow
\text{temporal action dependency}
\rightarrow
\text{repair localization}.
$$

## Baseline interpretation

The comparison baseline is deliberately strong: exact full uniform-cost
replanning from the updated initial state.

Cortex did **not** outperform that baseline in solution quality; it matched the
minimum cost in every feasible case.

Its advantage in the local-change fixtures was structural reuse:

- preserve valid prefix;
- preserve valid suffix;
- search only the damaged region;
- avoid re-exploring unrelated early dead ends.

## Claim boundary

A defensible claim is:

> Within declared finite deterministic action worlds, Cortex can compute
> minimum-cost plans, maintain temporal support provenance, localize the first
> affected step after an action-model change, repair an optimal local segment
> while preserving valid surrounding structure, fall back to global replanning
> when local repair is impossible, and report infeasibility when no plan exists.

This benchmark does **not** establish:

- probabilistic planning;
- partial observability;
- continuous control;
- autonomous discovery of action dynamics;
- general asymptotic planning improvements;
- large-scale planning superiority;
- human-like strategic planning.

## Reproducibility

Implementation:

- research/planning_replanning.py
- research/planning_replanning_campaign.py
- research/planning_replanning_gate.py
- tests/test_planning_replanning.py

Frozen protocol:

- PLANNING_REPLANNING_V1_PROTOCOL.md

Aggregate result:

- benchmarks/results/planning_replanning_v1_summary.json

## Next Reasoning Map axis

The next independent planned axis is **open-world novelty attribution v1**.

A particularly useful follow-up integration experiment would then combine
novelty attribution with replanning: distinguish whether plan failure came from
noise, a new state/entity/relation, or a genuine transition-regime change before
deciding how much of the plan to revise.
