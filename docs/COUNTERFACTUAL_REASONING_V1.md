# Cortex Reasoning Map v1 — Structural Counterfactual Reasoning v1

Status: **passed official benchmark / research-only**.

Protocol: \`counterfactual-reasoning-v1\`.

Official run: GitHub Actions run \`35434536047\`, job \`105874861338\`.

## Question

Can Cortex learn intervention-sensitive directed influence and then reason about
multi-hop consequences and temporary structural changes to the learned world?

The benchmark deliberately isolates **structural causal influence reasoning**.

It does not yet test full unit-level structural-equation counterfactual
inference.

## Setup

Every world contains 8 anonymously relabeled nodes.

Seven world families were evaluated:

- chain;
- fork;
- collider;
- diamond;
- bridge;
- disconnected components;
- random DAG.

The official seed range was frozen before evaluation:

$$
100,\ldots,119.
$$

That produced:

$$
7\times 20 = 140
$$

official worlds.

Development and preflight used only seeds below 100.

## Evidence model

The research prototype mirrors the native Cortex causal evidence thresholds:

- minimum intervention support: 8;
- minimum control support: 8;
- positive causal difference threshold: 0.24;
- resolved-null absolute difference threshold: 0.10.

The evidence fixtures are deterministic **controlled local direct-effect
probes**.

This is important.

v1 is not a noisy causal-discovery benchmark. It deliberately removes
estimation variance so that the experiment tests whether Cortex can reason over
a learned epistemic causal graph once the direct evidence is sufficiently
clear.

Indirect influence is never inserted directly into the pairwise evidence
cells. It must be recovered by graph composition.

## Epistemic reasoning rule

Cortex maintains two interpretations of the direct graph:

- the **confirmed graph**, containing only resolved causal edges;
- the **possible graph**, containing resolved causal and unresolved edges.

For source $s$ and target $t$:

$$
t\in\operatorname{Reach}_{confirmed}(s)
\Rightarrow
\text{affected},
$$

otherwise

$$
t\in\operatorname{Reach}_{possible}(s)
\Rightarrow
\text{unresolved},
$$

otherwise

$$
\text{unaffected}.
$$

This prevents lack of confirmed evidence from being silently converted into a
negative claim.

## Counterfactual operation

For a temporary edge-removal set $E_r$, Cortex reasons inside

$$
G' = G\setminus E_r.
$$

The base causal model is not mutated.

Every counterfactual query compares a deterministic state fingerprint before
and after evaluation.

## Official result

All frozen gates passed.

| capability | Cortex |
|---|---:|
| direct resolved precision | **1.000** |
| direct resolved coverage | **1.000** |
| causal-edge recall | **1.000** |
| null-edge recall | **1.000** |
| correlation-trap rejection | **1.000** |
| multi-hop accuracy | **1.000** |
| multi-hop coverage | **1.000** |
| multi-hop resolved accuracy | **1.000** |
| negative-query accuracy | **1.000** |
| counterfactual accuracy | **1.000** |
| counterfactual coverage | **1.000** |
| counterfactual resolved accuracy | **1.000** |
| redundant-path preservation | **1.000** |
| bridge-cut success | **1.000** |
| missing-bridge unresolved rate | **1.000** |
| valid proof-path rate | **1.000** |
| state-restoration rate | **1.000** |

The direct-edge baseline obtained multi-hop accuracy **0.000** because the
tested source-target pairs deliberately had no direct edge.

The correlation baseline obtained trap accuracy **0.000** because it treated a
high control outcome rate as causal even when intervention produced no change.

## What Cortex demonstrated

Within these fixtures, Cortex:

1. reconstructed the declared direct causal structure from
   intervention/control evidence;
2. composed direct edges into unseen multi-hop influence;
3. preserved directionality on negative queries;
4. evaluated causal consequences after temporary edge deletion;
5. preserved influence when an alternate diamond path remained;
6. removed influence when a bridge edge was cut;
7. returned unresolved when the only bridge was below the evidence threshold;
8. attached a valid confirmed path to every affected reasoning result;
9. restored the exact base-state fingerprint after every counterfactual query.

The most important qualitative result is not the perfect score itself.

It is the combination:

$$
\text{temporary world modification}
+
\text{reasoning inside the modified world}
+
\text{epistemic abstention}
+
\text{exact restoration}.
$$

That is a reasoning primitive we had not tested in the algebra campaigns.

## Why the perfect result should be interpreted cautiously

The direct evidence in v1 is intentionally deterministic and highly separated.

Therefore the campaign does **not** establish robust causal discovery under
noise.

A more realistic benchmark would need to vary:

- finite-sample noise;
- near-threshold causal effects;
- contradictory interventions;
- hidden confounders;
- cyclic systems;
- uncertain edge direction;
- changing causal mechanisms.

Those belong in later Reasoning Map campaigns.

## Claim boundary

A defensible claim is:

> Within the declared finite causal fixtures, Cortex demonstrated structural
> counterfactual influence reasoning: it learned intervention-sensitive direct
> structure, composed unseen multi-hop influence, reasoned under temporary edge
> removals, preserved redundant paths, abstained on unresolved bridges, and
> restored its base model exactly.

Do not claim from this benchmark alone:

- full Pearlian counterfactual reasoning;
- hidden-confounder identification;
- general causal discovery;
- real-world causal inference;
- superiority over causal graphical-model systems.

## Reproducibility

Implementation:

- \`research/counterfactual_reasoning.py\`
- \`research/counterfactual_reasoning_campaign.py\`
- \`research/counterfactual_reasoning_gate.py\`
- \`tests/test_counterfactual_reasoning.py\`

Frozen protocol:

- [\`COUNTERFACTUAL_REASONING_V1_PROTOCOL.md\`](COUNTERFACTUAL_REASONING_V1_PROTOCOL.md)

Aggregate result:

- [\`benchmarks/results/counterfactual_reasoning_v1_summary.json\`](../benchmarks/results/counterfactual_reasoning_v1_summary.json)

## Next Reasoning Map experiment

The next planned axis is **belief revision under contradiction**.

That campaign should test whether a strongly supported structural belief can
move through

$$
\text{active}
\rightarrow
\text{uncertain}
\rightarrow
\text{retracted}
$$

as contradictory evidence accumulates, and whether Cortex can later recover
when the regime changes again.
