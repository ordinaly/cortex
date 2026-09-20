# Cortex Reasoning Map v1 — Novelty-Aware Replanning v1

Status: **frozen protocol / official campaign passed**.

Protocol ID: novelty-aware-replanning-v1.

## Research question

Can Cortex use the **kind of attributed novelty** to decide how much of its
planning state should be revised?

The integration chain is:

$$
\text{surprise}
\rightarrow
\text{novelty attribution}
\rightarrow
\text{localized model edit}
\rightarrow
\text{planning dependency impact}
\rightarrow
\text{preserve / repair / restart / defer}.
$$

This milestone composes the already passed
open-world-novelty-attribution-v1 and planning-replanning-v1 primitives.

## Policy under test

The policy is frozen as follows.

- **noise**: preserve model and plan; perform no planning search;
- **new instance**: apply the instance binding, then repair only if the new
  instance affects an indexed planning opportunity or dependency;
- **new relation**: apply the relation edit, then repair only if that exact
  relation affects the plan;
- **new entity type**: apply the ontology edit, then repair from the first
  type-dependent planning point;
- **new regime**: treat transition assumptions as globally affected and perform
  full replanning from the initial state;
- **unresolved**: do not mutate the structural model, do not replan, and mark
  execution as deferred.

A resolved structural novelty therefore does **not** automatically imply plan
revision.

## Planning dependency index

Planning dependencies are represented explicitly.

Each action may carry structural dependencies. In addition, a newly learned
structural fact may unlock an alternative from an indexed trajectory position.

The integration layer maps a novelty edit onto dependency references:

- new instance -> instance ID;
- new entity type -> new type ID plus newly bound instance IDs;
- new relation -> exact typed relation key;
- new regime -> global regime dependency.

The earliest matching dependency or opportunity becomes the local impact index.

Wildcard type dependencies are allowed for actions whose availability depends
on any newly introduced type.

## Dependency-indexed repair

The original replanner detects changed or removed actions on the active path.

This integration adds one research helper for the complementary case: a newly
available action may improve the plan even though every old action remains
valid.

Given declared impact index k, Cortex:

1. verifies that the prefix before k remains exactly valid;
2. searches from the old state at k;
3. rejoins the earliest still-valid old trajectory state;
4. preserves the unaffected suffix;
5. falls back to full replanning if no valid local rejoin exists.

The same minimum-cost uniform-cost search remains the search oracle.

## Benchmark families

Eight families are frozen.

### noise-preserve

Novelty source: noise-spike.

Expected behavior:

- attribution = noise;
- decision = preserve;
- exact plan fingerprint preserved;
- zero planning expansions;
- no structural model edit.

### new-instance-irrelevant

Novelty source: new-instance.

The instance binding is structurally accepted but does not affect any active
planning dependency.

Expected behavior:

- attribution = new_instance;
- instance model component changes;
- active plan remains exact;
- zero planning expansions.

### new-instance-local

Novelty source: new-instance.

The newly bound instance unlocks a cheaper action at old trajectory index 2.

Expected behavior:

- attribution = new_instance;
- local repair from index 2;
- prefix before index 2 preserved;
- old suffix preserved after rejoin;
- repaired cost equals full-replan optimum;
- local search expands fewer states than full restart.

### new-relation-local

Novelty source: new-relation.

The exact newly learned typed relation unlocks a cheaper action at trajectory
index 3.

Expected behavior:

- relation-only structural edit;
- local repair from index 3;
- optimal cost equal to full replanning;
- lower search work than full restart.

### new-relation-irrelevant

Novelty source: new-relation.

The relation edit is real but unrelated to the active plan.

Expected behavior:

- relation model changes;
- plan fingerprint remains exact;
- zero planning expansions.

### new-type-local

Novelty source: new-entity-type.

The newly introduced type unlocks a cheaper type-dependent action at trajectory
index 1.

Expected behavior:

- type + new-instance binding edit;
- local repair from index 1;
- optimal cost equal to full replanning;
- preserved unaffected prefix/suffix.

### new-regime-global

Novelty source: new-regime.

The regime changes transition assumptions broadly: one former route action is
removed and a new global route becomes available.

Expected behavior:

- regime + active-regime model edit;
- global replanning from index 0;
- no claim of local suffix preservation;
- minimum cost equal to full replanning.

### unresolved-defer

Novelty source: ambiguous-instance-type.

A hypothetical ontology-dependent shortcut exists in the candidate updated
world, but the novelty attribution itself is unresolved.

Expected behavior:

- no structural edit;
- no planning search;
- decision = deferred;
- safe_to_execute = false.

The candidate updated planning world must not be adopted while attribution is
unresolved.

## Planning worlds

Every fixture uses an opaque six-step base chain with action cost 3.

Several cheap dead-end branches are reachable only from the initial state.
These branches make full restart measurably more expensive in state expansions
than local repair while remaining irrelevant to the optimal solution.

All state and action IDs are independently randomized per seed.

## Strong references

For every updated planning world, exact uniform-cost replanning from the initial
state is used as the solution-quality reference.

Local repair is not allowed to trade away plan cost for locality.

For every local-repair family:

$$
C_{local}=C_{full}.
$$

Search efficiency is measured separately in expanded states.

## Baselines

Two frozen baselines are used.

### Always-replan baseline

- preserve only noise;
- defer unresolved novelty;
- after every resolved structural novelty, restart exact planning globally.

This baseline should lose on revision locality and unnecessary search.

### Ignore-novelty baseline

Never apply the novelty edit and always keep the original plan.

This baseline should lose optimality or validity whenever a planning-relevant
novelty occurs.

## Development and official seeds

Development seeds:

0 ... 9

Untouched official seeds:

1600 ... 1619

With eight families, the official campaign contains **160 cases**.

The official range must remain untouched until the development campaign and
frozen gates pass unchanged.

## Frozen gates

### Attribution and decision

1. novelty attribution accuracy = **1.000**;
2. novelty-to-replanning decision accuracy = **1.000**;
3. returned-plan validity / deferred-plan validity = **1.000**;
4. structural model-component locality = **1.000**.

### Solution quality

5. local/global repair minimum-cost agreement with full replanning = **1.000**.

### Preservation

6. exact plan preservation on noise and irrelevant structural novelty = **1.000**;
7. zero planning search on preserve cases = **1.000**.

### Local repair

On new-instance-local, new-relation-local, and new-type-local:

8. exact impact-index localization = **1.000**;
9. exact unaffected-prefix preservation = **1.000**;
10. at least one old suffix action preserved = **1.000**;
11. local search expands fewer states than full replanning in every case;
12. mean local/full expansion ratio <= **0.50**.

### Global and unresolved controls

13. new-regime global-replan correctness = **1.000**;
14. unresolved deferral/no-edit/no-search safety = **1.000**.

### Baseline discrimination

15. always-replan decision accuracy <= **0.50**;
16. ignore-novelty optimality on planning-relevant novelty <= **0.25**.

No gate may be weakened after official results are observed.

## Negative-result policy

Development may fix implementation or fixture errors before any official run,
provided the scientific policy and frozen thresholds above remain unchanged.

If the untouched official campaign fails, preserve that result. Any revised
official claim requires a new protocol version and fresh untouched seeds.

## Claim boundary

A passing result would support:

> Within declared finite structured planning worlds, Cortex can use a resolved
> novelty attribution to determine whether to preserve an existing plan,
> perform dependency-local repair, restart planning globally under regime
> change, or defer revision under unresolved novelty, while retaining exact
> solution cost against a full uniform-cost planning reference.

It would not establish:

- real-world autonomous planning;
- learned action-model synthesis from raw observations;
- stochastic or partially observable planning;
- compound simultaneous novelty;
- adversarial novelty robustness;
- safe physical action under uncertainty;
- general asymptotic planning improvements.

## Planned files

- research/novelty_aware_replanning.py
- research/novelty_aware_replanning_campaign.py
- research/novelty_aware_replanning_gate.py
- tests/test_novelty_aware_replanning.py

The official campaign must remain untouched until development preflight is
green.


## Official outcome

The untouched official campaign on seeds **1600–1619** passed every frozen
gate across **160** cases.

Official metrics:

- novelty attribution accuracy: **1.000**;
- novelty-to-replanning decision accuracy: **1.000**;
- returned/deferred plan validity: **1.000**;
- structural model-component locality: **1.000**;
- repair optimal-cost agreement: **1.000**;
- exact preserve-case plan stability: **1.000**;
- zero-search preservation: **1.000**;
- local impact-index accuracy: **1.000**;
- local prefix/suffix preservation: **1.000 / 1.000**;
- local search-saving rate: **1.000**;
- mean local/full expansion ratio: **0.03846**;
- regime-global correctness: **1.000**;
- unresolved safety/deferral: **1.000**;
- always-replan decision accuracy: **0.375**;
- ignore-novelty optimality on affected cases: **0.000**.

See [NOVELTY_AWARE_REPLANNING_V1.md](NOVELTY_AWARE_REPLANNING_V1.md)
and
[benchmarks/results/novelty_aware_replanning_v1_summary.json](../benchmarks/results/novelty_aware_replanning_v1_summary.json).

The frozen policy and gates above were not changed after official evaluation.
