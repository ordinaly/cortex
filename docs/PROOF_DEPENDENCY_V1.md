# Cortex Reasoning Map v1 — Explicit Proof Dependency v1

Status: **passed official benchmark / research-only**.

Protocol: \`explicit-proof-dependency-v1\`.

Official run: GitHub Actions run \`35510159225\`, job \`106076541674\`.

## Question

Can Cortex keep multiple explicit justifications for one conclusion and revise
the conclusion correctly when individual premises are removed or restored?

The benchmark moves beyond carrying one valid reasoning path.

It maintains an explicit finite proof/dependency DAG in which each derived proof
records:

- parent proof IDs;
- source rule;
- transitive fact dependencies;
- transitive rule dependencies;
- derivation depth.

## Official campaign

Untouched official seeds:

$$
1300,\ldots,1319.
$$

Six proof-structure families per seed produced:

$$
20\times6=120
$$

official cases.

Atom labels, fact IDs, and rule IDs were independently opaque per seed.

## Result

Every frozen gate passed.

| metric | result |
|---|---:|
| active proof-DAG validity | **1.000** |
| rule/proof provenance validity | **1.000** |
| shortest selected-proof correctness | **1.000** |
| exact restoration after premise restoration | **1.000** |
| independent alternate-proof survival | **1.000** |
| shallow-to-deeper proof takeover | **1.000** |
| shared-root invalidation correctness | **1.000** |
| contradictory-proof conflict handling | **1.000** |
| nested partial invalidation correctness | **1.000** |
| irrelevant-premise locality | **1.000** |
| single-proof baseline failure on alternate-support cases | **1.000** |

The final line is intentional: the baseline fails in every designated case in
which it commits to one proof and discards a surviving independent derivation.

## Redundant support

In the redundant-branches family, two fact-disjoint derivations support the
same conclusion.

When one fact belonging to the currently selected proof is retracted:

- that proof disappears;
- every proof depending on the removed fact disappears;
- the independent proof retains the same proof ID;
- the target remains supported;
- the single-proof baseline becomes unresolved.

This passed on all 20 official seeds.

## Shortest versus merely valid proof

The short-vs-long family contains one shallow proof and one deeper independent
proof.

Before retraction, Cortex selects the shallow proof.

After one of its premises is removed, Cortex does not retract the conclusion.
The longer surviving proof becomes selected.

Deeper-proof takeover rate:

$$
1.000.
$$

This demonstrates an important distinction:

$$
\text{selected proof}
\neq
\text{only proof}.
$$

## Shared dependency

The shared-root family contains two apparently separate branches that depend on
one common source premise.

Removing that premise invalidated both branches in every official case.

So Cortex did not confuse multiple branch shapes with independent evidence.

## Contradictory proofs

The contradictory family simultaneously derives a target and its negation.

Required behavior was:

$$
P^+\neq\varnothing
\quad\text{and}\quad
P^-\neq\varnothing
\Rightarrow
\text{unresolved conflict}.
$$

This held in every official case.

Removing negative support produced **supported**.

Restoring it returned the state to **conflict**.

Removing positive support instead produced **refuted**.

No arbitrary tie-break was used.

## Nested invalidation

The nested-redundancy family contains alternative proofs for an intermediate
literal plus an independent final branch.

Retracting one nested source fact removed only the dependent proof subset.
Multiple target proofs remained active and the target stayed supported.

Nested partial-invalidation correctness:

$$
1.000.
$$

## Locality

In the irrelevant-premise family, a fact belongs only to a separate proof
branch that never reaches the target.

Retracting that fact left the complete target explanation fingerprint
unchanged on every official seed.

This is semantic dependency locality: unrelated proof structure does not
rewrite the target explanation.

## Exact restoration

After every temporary fact retraction, restoring the fact reproduced the exact
target explanation fingerprint.

Restoration rate:

$$
1.000.
$$

## What this adds to Cortex

Earlier counterfactual reasoning attached one valid path to a resolved
conclusion.

This milestone introduces a stronger representation:

$$
\text{conclusion}
\longleftrightarrow
\{\text{all active finite proof alternatives}\}.
$$

A conclusion can therefore survive damage to one justification, become
unresolved when support genuinely conflicts, or disappear when its last
dependency is removed.

This is directly useful for later planning/replanning: an action or plan step
can depend on explicit conclusions, and a local premise change can identify
which downstream justifications actually need repair.

## Claim boundary

A defensible claim is:

> Within declared finite acyclic rule fixtures, Cortex can maintain explicit
> proof dependencies, preserve independent alternative derivations, invalidate
> exactly the proofs affected by premise removal, switch from a shortest proof
> to a longer surviving proof, represent positive/negative proof conflict as
> unresolved, restore explanations exactly, and preserve unrelated proof
> structure.

This benchmark does **not** establish:

- unrestricted theorem proving;
- discovery of inference rules;
- soundness/completeness for arbitrary formal logics;
- scalable search across very large theorem spaces;
- probabilistic confidence calibration of proofs.

The rules are supplied. The benchmark isolates proof dependency management.

## Reproducibility

Implementation:

- \`research/proof_dependency.py\`
- \`research/proof_dependency_campaign.py\`
- \`research/proof_dependency_gate.py\`
- \`tests/test_proof_dependency.py\`

Frozen protocol:

- [PROOF_DEPENDENCY_V1_PROTOCOL.md](PROOF_DEPENDENCY_V1_PROTOCOL.md)

Aggregate result:

- [\`benchmarks/results/proof_dependency_v1_summary.json\`](../benchmarks/results/proof_dependency_v1_summary.json)

## Next Reasoning Map axis

The next independent axis is **planning / replanning v1**.

The key Cortex-specific test is whether a local dependency or transition change
causes a local plan repair rather than a full restart.
