# Cortex Reasoning Map v1

Status: **active research program**.

The Reasoning Map is a capability-oriented benchmark program for Cortex.

Its goal is not to produce one aggregate "intelligence score." Instead it asks
which distinct reasoning operations Cortex can perform, how they degrade, and
where its explicit unresolved state provides useful epistemic restraint.

## Capability map

| capability | benchmark | status | strongest current result |
|---|---|---|---|
| recursive typed composition | \`kinship-composition-v1\` | passed | 1.000 accuracy / 1.000 coverage on unseen depth 6–10 |
| candidate-law induction | \`algebra-induction-v1\` | passed | exact held-out recovery when supported laws were evidenced |
| bounded law-form discovery | \`algebra-form-induction-v1.7\` | passed | 375-form grammar; zero wrong resolved predictions in 120 cases |
| cross-world abstract transfer | \`algebra-transfer-v1.6\` | passed | +0.320 coverage over scratch at 20% target evidence; safe negative transfer |
| structural counterfactual reasoning | \`counterfactual-reasoning-v1\` | passed | all frozen gates passed across 140 worlds |
| belief revision under contradiction | `belief-revision-v1` | passed | retraction 7/14/21 batches for prior strengths 1/2/3; exact locality |
| structural analogy / isomorphism | planned | next | — |
| latent concept induction | planned | queued | — |
| explicit proof dependency | planned | later | — |
| planning / replanning | planned | later | — |
| open-world novelty attribution | planned | later | — |

All passed entries are research fixtures, not stable runtime semantics unless
separately promoted through the executable-specification process.

## Why these axes are separated

A system that performs well on algebraic closure may still fail at causal
revision, analogy, planning, or concept invention.

The Reasoning Map therefore treats the following as distinct operations:

$$
\text{deduction},
\quad
\text{induction},
\quad
\text{transfer},
\quad
\text{counterfactual modification},
\quad
\text{revision},
\quad
\text{analogy},
\quad
\text{latent abstraction}.
$$

Each benchmark should isolate one or two primitives before later campaigns
combine them.

## Completed axis: structural counterfactual reasoning

\`counterfactual-reasoning-v1\` tested a new reasoning primitive:

$$
G
\rightarrow
G'
\rightarrow
\text{reason inside }G'
\rightarrow
G.
$$

The base graph is learned from intervention/control evidence. A
counterfactual query removes an edge only inside an ephemeral world
interpretation.

Across 140 untouched official worlds:

- direct causal resolved precision: **1.000**;
- multi-hop resolved accuracy: **1.000**;
- counterfactual resolved accuracy: **1.000**;
- redundant-path preservation: **1.000**;
- bridge-cut success: **1.000**;
- missing-bridge unresolved rate: **1.000**;
- proof-path validity: **1.000**;
- exact state restoration: **1.000**.

The evidence fixtures were intentionally deterministic controlled direct-effect
probes. The result therefore establishes the structural reasoning primitive,
not robust real-world causal discovery.

See [COUNTERFACTUAL_REASONING_V1.md](COUNTERFACTUAL_REASONING_V1.md).

## Completed axis: belief revision under contradiction

\`belief-revision-v1\` tested cumulative revision of one causal belief inside the
only path $A\rightarrow B\rightarrow C$.

Every frozen gate passed across 60 untouched official cases.

The central latency result was:

| initial support | retraction latency | recovery latency |
|---:|---:|---:|
| 1 batch | **7** | **3** |
| 2 batches | **14** | **5** |
| 3 batches | **21** | **7** |

Cortex resisted one contradictory batch, passed through unresolved before
retraction, remained unresolved under weaker ambiguous contradiction, recovered
when causal evidence returned, propagated each epistemic state downstream, and
left every unrelated evidence cell unchanged.

See [BELIEF_REVISION_V1.md](BELIEF_REVISION_V1.md).

## Planned axis: structural analogy

The analogy benchmark should test whether Cortex recognizes structural
equivalence across worlds with different:

- entity labels;
- relation labels;
- superficial graph layouts.

The central object is an isomorphism or partial structural correspondence:

$$
\phi:G_1\rightarrow G_2.
$$

The test should measure whether a structure learned in $G_1$ reduces evidence
requirements in $G_2$ without forcing transfer when the analogy breaks.

## Planned axis: latent concept induction

The latent-concept benchmark removes one more piece of supplied structure.

Instead of providing every useful category, Cortex should be rewarded for
introducing a latent grouping only when it reduces unresolved predictive
complexity.

Conceptually:

$$
\{e_1,e_3,e_7\}
\rightarrow
Z
$$

should be introduced only when the new articulation earns its complexity cost.

This is likely one of the strongest future tests of Cortex's core design
principle:

> Explain new evidence with the smallest structural change that future evidence
> can justify.

## Common evaluation axes

As the Reasoning Map expands, the same difficulty axes should be swept across
multiple benchmark families:

- reasoning depth;
- missing evidence;
- noise;
- contradiction rate;
- entity count;
- rule count;
- distribution shift;
- memory pressure.

This will produce degradation curves rather than isolated pass/fail scores.

## Confidence and abstention

Future Reasoning Map campaigns should report not only correctness but the
quality of Cortex's epistemic behavior.

Important questions include:

- Does confidence fall before errors rise?
- Does unresolved activate near structural ambiguity?
- Are high-confidence conclusions actually better calibrated?
- Does Cortex preserve multiple hypotheses when evidence cannot distinguish
  them?

Raw accuracy alone is not sufficient.

## Comparison policy

When a reasoning axis stabilizes, compare the same inputs and information
budget against appropriate baselines such as:

- direct memory;
- hand-engineered symbolic closure;
- Bayesian models;
- SAT/SMT or theorem-proving systems where relevant;
- graph neural networks;
- Transformers.

The purpose is a **reasoning profile**, not an overall winner.

## Claim discipline

The Reasoning Map can support capability-specific statements tied to declared
fixtures.

It must not be collapsed into a claim of AGI, universal reasoning, or general
superiority to neural or symbolic methods.
