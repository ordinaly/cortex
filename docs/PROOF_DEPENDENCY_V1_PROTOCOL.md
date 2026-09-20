# Cortex Reasoning Map v1 — Explicit Proof Dependency v1

Status: **frozen protocol / official campaign untouched**.

Protocol ID: \`explicit-proof-dependency-v1\`.

## Research question

Can Cortex retain **multiple explicit justifications** for the same conclusion
and revise only the conclusions/proofs that actually depend on a changed
premise?

The intended step beyond counterfactual-reasoning-v1 is:

$$
\text{one valid path}
\longrightarrow
\text{explicit proof/dependency DAG}.
$$

The benchmark tests dependency bookkeeping and epistemic revision over finite
supplied rules.

It does **not** test unrestricted theorem discovery.

## Proof representation

A signed literal is:

$$
\ell=(a,s),
\qquad
s\in\{-1,+1\}.
$$

A fact is a source proof node for one literal.

A rule has finite premises and one signed conclusion:

$$
\ell_1,\ldots,\ell_k
\Rightarrow
\ell.
$$

Every derived proof node stores:

- its conclusion literal;
- the rule that produced it;
- parent proof-node IDs;
- transitive fact dependencies;
- transitive rule dependencies;
- proof depth.

All rule dependencies must form an acyclic literal graph.

For a target atom $g$, Cortex retains every active positive proof of $g$ and
every active negative proof of $g$ within the declared finite proof bound.

## Epistemic state

For target atom $g$:

- positive proofs only → **supported**;
- negative proofs only → **refuted**;
- both positive and negative proofs → **conflict / unresolved**;
- neither side → **unproven / unresolved**.

If $P^+(g)$ and $P^-(g)$ are the active proof sets:

$$
P^+(g)\neq\varnothing,\quad P^-(g)=\varnothing
\Rightarrow +1,
$$

$$
P^-(g)\neq\varnothing,\quad P^+(g)=\varnothing
\Rightarrow -1,
$$

and otherwise Cortex returns state $0$.

No arbitrary conflict tie-break is allowed.

## Selected proof

Cortex may expose one selected proof for compact use, but selection does not
delete alternatives.

The selected proof is the lexicographically stable minimum under:

1. derivation depth;
2. number of fact dependencies;
3. number of rule dependencies;
4. stable proof ID.

Thus the selected proof is shortest among active derivations, while longer
valid derivations remain represented.

## Retraction and restoration

Retracting a fact removes exactly the proof nodes whose transitive dependency
set contains that fact.

Proofs with no dependency on the removed fact must retain stable proof IDs.

Restoring the fact must reconstruct the exact target explanation fingerprint.

This benchmark evaluates semantic dependency locality, not incremental runtime
complexity: the research implementation may rebuild the finite proof DAG after
a fact-state change.

## Families

Six structural families are frozen.

### redundant-branches

Two fact-disjoint proof branches derive the same positive conclusion.

Removing a premise from the currently selected branch must invalidate that
proof while preserving the independent alternate proof and the conclusion.

A single-proof baseline must fail this case.

### short-vs-long

A shallow proof and a deeper independent proof both support the target.

Initially the shallow proof must be selected.

After one of its premises is removed, the deeper valid proof must take over
without retracting the conclusion.

### shared-root

Two visually separate proof branches share one hidden source premise.

Removing the shared premise must invalidate **both** branches.

This distinguishes true proof independence from branch-counting.

### contradictory

One proof derives the positive target and another derives its negation.

With both active, Cortex must report conflict/unresolved.

Removing the negative support must yield supported.

Removing the positive support instead must yield refuted.

Restoring both must return to conflict.

### nested-redundancy

One intermediate literal has two alternative derivations and the final target
also has an independent branch.

Removing one nested premise must invalidate only the dependent proof subset
while leaving multiple target proofs active.

### irrelevant-premise

An unrelated fact supports a separate branch that never reaches the target.

Retracting that fact must leave the complete target explanation fingerprint
unchanged.

## Opaque naming

For every seed:

- atom labels are randomized opaque tokens;
- fact IDs are randomized opaque tokens;
- rule IDs are randomized opaque tokens.

The benchmark therefore does not depend on canonical names such as "goal",
"left", or "negative".

## Baseline

The baseline commits to the initially selected positive proof and discards all
alternatives.

After a premise of that selected proof is removed, the baseline returns
unresolved even when an independent alternate proof still supports the
conclusion.

The required discrimination cases are:

- redundant-branches;
- short-vs-long.

## Development and official seeds

Development seeds:

$$
0,\ldots,9.
$$

Untouched official seeds:

$$
1300,\ldots,1319.
$$

With six families, the official campaign contains:

$$
20\times6=120
$$

cases.

The official seed range must not run until the development campaign and frozen
gates pass unchanged.

## Frozen gates

The campaign passes only if every following rate equals **1.000**, except the
single-proof baseline which must fail every designated alternate-support case.

### Structural validity

1. active proof DAG validity = **1.000**;
2. rule/proof provenance validity = **1.000**;
3. selected-proof shortestness = **1.000**;
4. exact explanation restoration after fact restoration = **1.000**.

### Dependency behavior

5. independent alternate-proof survival = **1.000**;
6. exact invalidation of proofs depending on the removed premise = **1.000**;
7. shallow-to-deeper proof takeover = **1.000**;
8. shared-root full invalidation = **1.000**;
9. contradictory-proof conflict handling = **1.000**;
10. nested partial invalidation = **1.000**;
11. irrelevant-premise locality = **1.000**.

### Baseline discrimination

12. on redundant-branches and short-vs-long, Cortex retains the supported
    conclusion after selected-proof damage while the single-proof baseline
    becomes unresolved in **100%** of cases.

No threshold may be weakened after official results are observed.

## Negative-result policy

If development reveals an implementation/protocol mismatch, fix it before any
official run while keeping the declared scientific requirements unchanged.

If the untouched official campaign fails, preserve the result. Any revised
official claim must use a new protocol version and fresh untouched seeds.

## Claim boundary

A passing benchmark would support the following narrow claim:

> Within declared finite acyclic rule fixtures, Cortex can maintain explicit
> proof dependencies, preserve multiple independent derivations, invalidate
> only proofs whose premises are retracted, fall back from a shortest proof to
> a longer surviving proof, expose contradictory proof support as unresolved,
> and preserve unrelated explanations.

It would not establish:

- unrestricted theorem proving;
- automatic discovery of inference rules;
- formal proof assistant soundness for arbitrary logics;
- probabilistic proof calibration;
- scalable proof search in very large rule systems.

## Planned files

- \`research/proof_dependency.py\`
- \`research/proof_dependency_campaign.py\`
- \`research/proof_dependency_gate.py\`
- \`tests/test_proof_dependency.py\`

The official campaign must remain untouched until development preflight is
green.
