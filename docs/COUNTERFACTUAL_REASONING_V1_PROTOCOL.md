# Cortex Reasoning Map v1 — Structural Counterfactual Reasoning

Status: **frozen protocol / official campaign passed**.

Protocol ID: `counterfactual-reasoning-v1`.

This is the first experiment in the broader **Cortex Reasoning Map v1**
campaign. It follows the passing algebraic form-discovery and cross-world
transfer milestones.

## Research question

Can Cortex learn a directed causal-influence structure from explicit
intervention/control evidence and then reason correctly about:

1. unseen multi-hop interventions;
2. temporary structural edits to that learned world;
3. redundant causal paths;
4. missing causal evidence;
5. the distinction between correlation and intervention;
6. restoration of the original world after a counterfactual query?

The intended reasoning pattern is:

$$
\text{intervention evidence}
\rightarrow
\text{causal structure}
\rightarrow
\text{multi-hop consequence}
\rightarrow
\text{temporary world edit}
\rightarrow
\text{counterfactual consequence}.
$$

## Scientific boundary

This benchmark tests **structural causal counterfactual reasoning**.

It does **not** yet claim full Pearlian unit-level counterfactual inference.
The v1 query asks whether a target is structurally influenced by an
intervention source under a learned directed graph, optionally after a
temporary edge edit.

The result must therefore be described as causal-structure / influence
reasoning, not as a complete structural-equation-model counterfactual engine.

## Native semantic alignment

The evidence model mirrors the stable native Cortex causal-evidence contract.

For each ordered pair $(i,j)$ Cortex stores interventional and control
Beta-style sufficient statistics:

$$
(\alpha_{do},\beta_{do}),
\qquad
(\alpha_{ctrl},\beta_{ctrl}).
$$

With posterior means

$$
p_{do}
=
\frac{\alpha_{do}}
     {\alpha_{do}+\beta_{do}},
\qquad
p_{ctrl}
=
\frac{\alpha_{ctrl}}
     {\alpha_{ctrl}+\beta_{ctrl}},
$$

define

$$
\Delta_{ij}
=
p_{do}-p_{ctrl}.
$$

The frozen native thresholds are retained:

- at least 8 intervention observations;
- at least 8 control observations;
- causal when $\Delta_{ij}\ge 0.24$;
- resolved-null when $|\Delta_{ij}|\le 0.10$;
- otherwise unresolved.

The research prototype is independent Python code, but the evidence-state
semantics are intentionally the same as the current Rust graph contract.

## Evidence fixtures

v1 isolates reasoning from stochastic-estimation variance, so evidence counts
are deterministic.

Each ordered-pair fixture is a **controlled local direct-effect probe**. Indirect
effects are intentionally not folded into the pairwise evidence cell; they are
reserved for the subsequent graph-composition query stage. This keeps direct
structure induction and multi-hop reasoning experimentally separable.

Every fully observed ordered pair receives 20 intervention and 20 control
observations.

### True direct causal edge

$$
n_{do}^{+}=18,\quad n_{do}^{-}=2,
$$

$$
n_{ctrl}^{+}=2,\quad n_{ctrl}^{-}=18.
$$

### Resolved null edge

$$
n_{do}^{+}=2,\quad n_{do}^{-}=18,
$$

$$
n_{ctrl}^{+}=2,\quad n_{ctrl}^{-}=18.
$$

### Correlation trap

For selected non-causal pairs:

$$
n_{do}^{+}=16,\quad n_{do}^{-}=4,
$$

$$
n_{ctrl}^{+}=16,\quad n_{ctrl}^{-}=4.
$$

These pairs have a high outcome rate but no intervention effect. Cortex should
resolve them as null rather than causal.

### Missing-evidence diagnostic

A selected true bridge edge receives only four intervention and four control
observations, below the frozen minimum support.

Its direct state must therefore remain unresolved.

## Epistemic graph semantics

Cortex constructs two directed graphs.

### Confirmed graph

Contains only direct edges whose evidence state is causal.

### Possible graph

Contains confirmed edges plus unresolved direct edges. Resolved-null edges are
excluded.

For a query source $s$ and target $t$:

1. if $t$ is reachable from $s$ in the confirmed graph, return **affected**;
2. otherwise, if $t$ is reachable in the possible graph, return
   **unresolved**;
3. otherwise return **unaffected**.

This is the key abstention rule:

$$
\text{no confirmed path}
\not\Rightarrow
\text{no possible path}.
$$

A missing bridge should therefore create uncertainty rather than a false
negative.

## Counterfactual structural edits

A query may supply a temporary set of directed edge removals.

For

$$
G' = G\setminus E_{remove},
$$

Cortex recomputes confirmed and possible reachability inside $G'$.

The edit is **ephemeral**. It must not mutate the learned base model.

Every counterfactual query records a deterministic state fingerprint before and
after evaluation. The fingerprints must be identical.

v1 tests removals only. Edge insertion/replacement and mechanism-level
counterfactuals are reserved for later campaigns.

## World families

Every world has 8 anonymously relabeled nodes.

The official campaign uses 20 deterministic seeds per family: **100–119**.
Preflight and unit tests use only development seeds below 100.

### Chain

A depth-5 chain plus isolated nodes.

Purpose:
- depth extrapolation;
- multi-hop influence.

### Fork

One source branches into multiple downstream chains.

Purpose:
- one-to-many causal influence.

### Collider

Two independent parents converge on a shared downstream node.

Purpose:
- directionality;
- avoid inventing reverse influence.

### Diamond

Two disjoint paths connect a source to a downstream target.

Purpose:
- redundant causal explanation;
- removing one edge must preserve downstream influence when the alternate path
  remains.

### Bridge

Two subgraphs are connected by one causal bridge.

Purpose:
- removing the bridge must eliminate confirmed downstream influence.

### Disconnected

Two independent causal components.

Purpose:
- reject cross-component influence.

### Random DAG

Eight nodes, a fixed topological order, and deterministic seed-specific sparse
edges.

Purpose:
- avoid overfitting to hand-designed motifs.

All node labels are independently permuted to opaque names for every seed.

## Correlation traps

Each world contains two non-causal ordered pairs with the high/high
intervention-control fixture above.

The pairs are selected from non-edges after world construction.

They are never treated as ground-truth causal edges.

## Query sets

### Direct-edge characterization

Score every ordered pair against the ground-truth direct DAG.

Metrics:
- resolved precision;
- resolved coverage;
- causal-edge recall;
- null-edge recall;
- correlation-trap rejection.

### Multi-hop intervention queries

Evaluate ordered source-target pairs connected by paths of length at least 2.

Direct source-target edges are excluded.

This forces composition over learned causal structure.

### Negative reachability queries

Evaluate pairs in opposite direction or across disconnected components.

### Edge-removal counterfactuals

For every eligible active edge, temporarily remove the edge and query selected
downstream targets.

Special tagged subsets:

- **redundant-path** cases: target should remain affected;
- **bridge-cut** cases: target should become unaffected when all relevant
  edges are resolved.

### Missing-bridge diagnostic

A separate bridge-world diagnostic leaves the true bridge below evidence
threshold while keeping all other pairwise evidence unchanged.

Queries whose only possible route requires that bridge should return
**unresolved**, not unaffected.

## Proof provenance

Every resolved affected answer must carry one confirmed directed path

$$
s\rightarrow\cdots\rightarrow t.
$$

The path must:

- begin at the query source;
- end at the target;
- use only currently active confirmed edges;
- exclude every temporarily removed edge.

This is not yet the dedicated proof-DAG benchmark, but it establishes basic
reasoning provenance.

## Baselines

### Direct-edge baseline

Predicts affected only when $(s,t)$ itself is a resolved causal edge.

It cannot compose multi-hop paths.

### Correlation baseline

Treats a pair as causal whenever the observed control outcome rate exceeds
0.5.

It is deliberately vulnerable to the correlation traps.

### Oracle closure

Uses the ground-truth DAG.

This is an upper-bound diagnostic, not a competitor.

## Official metrics

Per world:

- direct resolved precision;
- direct resolved coverage;
- direct causal recall;
- direct null recall;
- correlation-trap rejection;
- multi-hop accuracy;
- multi-hop coverage;
- multi-hop resolved accuracy;
- negative-query accuracy;
- counterfactual accuracy;
- counterfactual coverage;
- counterfactual resolved accuracy;
- redundant-path preservation rate;
- bridge-cut success rate;
- missing-bridge unresolved rate;
- valid-proof-path rate;
- state-restoration rate;
- direct-edge baseline multi-hop accuracy;
- correlation-baseline trap accuracy.

Campaign aggregates are macro-averaged across seeds and families unless
otherwise stated.

## Frozen gates

The official campaign passes only if all of the following hold.

### Causal-structure learning

1. direct resolved precision >= **0.995**;
2. direct resolved coverage >= **0.98**;
3. causal-edge recall >= **0.98**;
4. null-edge recall >= **0.98**;
5. correlation-trap rejection >= **0.99**.

### Multi-hop reasoning

6. multi-hop resolved accuracy >= **0.995**;
7. multi-hop coverage >= **0.98**;
8. Cortex multi-hop accuracy exceeds the direct-edge baseline by at least
   **0.25**.

### Negative influence

9. negative-query accuracy >= **0.995**.

### Structural counterfactuals

10. counterfactual resolved accuracy >= **0.995**;
11. counterfactual coverage >= **0.98**;
12. redundant-path preservation >= **0.99**;
13. bridge-cut success >= **0.99**.

### Epistemic uncertainty

14. missing-bridge unresolved rate >= **0.99**.

### Provenance and reversibility

15. valid proof-path rate = **1.0**;
16. state-restoration rate = **1.0**.

### Correlation control

17. correlation baseline trap accuracy <= **0.10**.

No threshold may be weakened after official campaign results are observed.

## Negative-result policy

Any failed gate is retained.

If a preflight exposes an implementation mismatch with this protocol, the
protocol version must advance and the revision must be documented before any
official campaign is run.

If an official campaign fails, that result remains part of the research
record. A later protocol must use fresh untouched seeds for any new official
claim.

## Planned files

- `research/counterfactual_reasoning.py`
- `research/counterfactual_reasoning_campaign.py`
- `research/counterfactual_reasoning_gate.py`
- `tests/test_counterfactual_reasoning.py`

The official campaign must not be run until unit preflight is green.

## Intended claim boundary

A passing result would support:

> Within the declared finite causal fixtures, Cortex can learn directed
> intervention-sensitive influence, compose it over unseen multi-hop queries,
> evaluate temporary edge-removal counterfactuals, preserve redundant paths,
> abstain when a required bridge is unresolved, and restore its base model
> exactly afterward.

It would not establish full causal discovery, hidden-confounder reasoning, or
unit-level structural-equation counterfactual inference.


## Official outcome

The untouched official campaign on seeds 100–119 passed all frozen gates
across 140 worlds.

See [COUNTERFACTUAL_REASONING_V1.md](COUNTERFACTUAL_REASONING_V1.md) and
[`benchmarks/results/counterfactual_reasoning_v1_summary.json`](../benchmarks/results/counterfactual_reasoning_v1_summary.json).

The frozen thresholds above were not weakened after evaluation.
