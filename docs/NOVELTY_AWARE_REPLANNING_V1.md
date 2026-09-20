# Cortex Reasoning Map v1 — Novelty-Aware Replanning v1

Status: **passed official benchmark / research-only**.

Protocol: novelty-aware-replanning-v1.

Official run: GitHub Actions run 35513405020, job 106085165477.

## Question

Can Cortex use the **kind of attributed novelty** to decide whether an active
plan should be preserved, repaired locally, restarted globally, or deferred?

This campaign composes two previously passed primitives:

- open-world novelty attribution;
- planning / replanning.

The tested loop is:

$$
\text{surprise}
\rightarrow
\text{novelty attribution}
\rightarrow
\text{localized structural edit}
\rightarrow
\text{planning dependency impact}
\rightarrow
\text{preserve / repair / restart / defer}.
$$

## Official campaign

Untouched official seeds **1600–1619** across eight structural families
produced **160 cases**.

Every frozen gate passed.

| metric | result |
|---|---:|
| novelty attribution accuracy | **1.000** |
| novelty-to-replanning decision accuracy | **1.000** |
| returned/deferred plan validity | **1.000** |
| structural model-component locality | **1.000** |
| repair optimal-cost agreement | **1.000** |
| exact preserve-case plan stability | **1.000** |
| zero-search preservation rate | **1.000** |
| local impact-index accuracy | **1.000** |
| local prefix preservation | **1.000** |
| local suffix preservation | **1.000** |
| local search-saving rate | **1.000** |
| mean local/full expansion ratio | **0.03846** |
| regime-global decision correctness | **1.000** |
| unresolved safety/deferral | **1.000** |
| always-replan decision accuracy | **0.375** |
| ignore-novelty optimality on affected cases | **0.000** |

## Main result

The system did not treat every structural surprise as a reason to restart
planning.

Instead:

- noise preserved both model and plan;
- irrelevant new instances and relations updated the structural model but left
  the plan fingerprint unchanged with zero planning search;
- planning-relevant new instances, relations, and entity types triggered
  dependency-local repair;
- regime change triggered global replanning;
- unresolved novelty triggered no structural edit, no search, and execution
  deferral.

This is the intended distinction between **model revision** and **plan
revision**.

## Local repair efficiency

Across the local-repair families, the mean ratio of Cortex local search
expansions to full replanning expansions was:

$$
0.03846.
$$

Equivalently, local repair expanded about **96.15% fewer states** on these
declared fixtures while matching the full uniform-cost planning optimum in
every case.

This is a fixture-specific state-expansion result, not a general complexity or
runtime theorem.

## Noise preservation

Noise produced:

- no structural model edit;
- exact plan preservation;
- zero planning expansions;
- safe execution of the unchanged plan.

This prevents the adaptive loop from turning ordinary residual noise into
planning churn.

## Resolved but irrelevant novelty

The campaign includes two explicit locality controls:

- a genuinely new instance unrelated to the active plan;
- a genuinely new relation unrelated to the active plan.

In both cases Cortex accepted the structural edit while preserving the exact
plan and performing zero planning search.

This is important because a policy based only on novelty category would be too
coarse. The plan changes only when the concrete novelty edit intersects an
explicit planning dependency or opportunity.

## Planning-relevant new instance

A newly bound instance unlocked a cheaper action at a declared dependency
point.

Cortex:

- accepted the instance binding;
- localized the first planning impact;
- repaired from that point;
- preserved the unaffected prefix;
- rejoined and preserved the old suffix;
- matched the global optimum;
- searched less than full replanning.

## Planning-relevant new relation

The exact newly learned typed relation unlocked a cheaper local action.

Cortex used the relation key itself to find the planning impact point rather
than globally resetting the plan.

The relation model changed; unrelated ontology components did not.

## Planning-relevant new entity type

A newly created type unlocked a type-dependent action.

The ontology edit remained local to:

- the type library;
- new-instance bindings.

Planning was repaired only from the type-dependent trajectory point.

## Regime change

The regime family deliberately changed transition assumptions broadly.

Cortex did not attempt to force locality. It performed full replanning from the
initial state and matched the exact full-search optimum.

This is the complementary behavior to local repair:

> preserve locality only while the structural assumptions supporting locality
> remain justified.

## Unresolved novelty

The ambiguous instance-vs-type evidence remained unresolved.

Cortex therefore:

- emitted no structural model edit;
- performed no planning search;
- preserved the previous plan as historical state;
- marked execution as **deferred / not safe to execute**.

The candidate updated planning world was not adopted.

This is the integration-level epistemic brake: unresolved ontology revision
does not silently propagate into irreversible action.

## Baselines

The always-replan baseline achieved only **0.375** decision accuracy.

It unnecessarily restarted planning for local or irrelevant novelty.

The ignore-novelty baseline had **0.000** optimality on planning-relevant
novelty cases.

It either retained a now-suboptimal plan or failed to adapt to the changed
transition model.

Together, the controls reject both extremes:

- revise everything;
- revise nothing.

## Architectural consequence

The research stack now supports the following explicit adaptive loop:

$$
\text{prediction}
\rightarrow
\text{surprise}
\rightarrow
\text{structural attribution}
\rightarrow
\text{model edit}
\rightarrow
\text{dependency localization}
\rightarrow
\text{plan response}.
$$

The response can be:

$$
\text{preserve},
\quad
\text{local repair},
\quad
\text{global replan},
\quad
\text{defer}.
$$

This is the first Cortex Reasoning Map campaign that composes several previously
validated reasoning primitives into one closed adaptation decision.

## Claim boundary

A defensible claim is:

> Within declared finite structured planning worlds, Cortex can use a resolved
> novelty attribution to determine whether to preserve an existing plan,
> perform dependency-local repair, restart planning globally under regime
> change, or defer revision under unresolved novelty, while retaining exact
> solution cost against a full uniform-cost planning reference.

This benchmark does **not** establish:

- real-world autonomous planning;
- learned action-model synthesis from raw observations;
- stochastic or partially observable planning;
- compound simultaneous novelty;
- adversarial novelty robustness;
- safe physical action under uncertainty;
- general asymptotic planning improvement.

## Reproducibility

Implementation:

- research/novelty_aware_replanning.py
- research/novelty_aware_replanning_campaign.py
- research/novelty_aware_replanning_gate.py
- research/planning_replanning.py
- tests/test_novelty_aware_replanning.py

Frozen protocol:

- [NOVELTY_AWARE_REPLANNING_V1_PROTOCOL.md](NOVELTY_AWARE_REPLANNING_V1_PROTOCOL.md)

Aggregate result:

- [benchmarks/results/novelty_aware_replanning_v1_summary.json](../benchmarks/results/novelty_aware_replanning_v1_summary.json)

## Recommended next milestone

The next scientific milestone should be **Open-World Robustness v1**.

Rather than add another clean deterministic primitive, it should stress the
combined adaptive loop with:

- stochastic corruption;
- missing evidence;
- persistence sweeps;
- gradual regime change;
- compound novelty;
- confidence calibration;
- revision/recovery latency;
- false-certainty and unnecessary-replan rates.

The central question becomes whether unresolved activates before incorrect
structural edits or incorrect plan revisions dominate.
