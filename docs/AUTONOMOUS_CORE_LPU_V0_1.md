# Autonomous Core + LPU v0.1

Status: **active architecture pivot / research prototype**.

## Decision

The primary Cortex path is now autonomous structural learning from chronological
observations. A neural network is no longer part of the main cognitive loop.

The former Ollama/Qwen language-gateway and provider-independent Hybrid Core
experiments remain recoverable from Git history and the preserved experimental
branch, but they are not dependencies of the autonomous main architecture.

The intended boundary is:

```text
structured observations
        |
        v
   Cortex memory
        |
        +----> structural learning / revision / abstraction
        |
        +----> temporary articulation ----> LPU
                                      exact local computation
```

External perceptual or language adapters may be studied later, but they must not
own Cortex memory, ontology, or reasoning policy.

## Direct dataset learning

`cortex.autonomy.ObservationFrame` is the first neutral ingestion contract.
`train_dataset(...)` presents frames to the native Cortex runtime in their
original chronological order and returns a final runtime snapshot.

This is intentionally closer to continual exposure than to neural-network
training:

- there are no gradients;
- there is no loss backpropagated into a model;
- observations update Cortex's own persistent state;
- later evidence remains free to revise earlier structure;
- dataset provenance can be carried without being interpreted as ontology.

That teacher-detachment benchmark has now been completed by **Cortex Autonomy Campaign v1**. It established teacher-free held-out inference, post-detachment transfer, local contradiction handling, and LPU use on the declared structured fixtures. The subsequent **Plasticity Frontier v1** exposed lifetime-history inertia, and **Relational Regime Plasticity v0.1** removed that measured proportional re-articulation debt while retaining durable history. The current integration target is **Autonomy Robustness & Temporal Plasticity v1**.

See [CORTEX_AUTONOMY_CAMPAIGN_V1.md](CORTEX_AUTONOMY_CAMPAIGN_V1.md), [PLASTICITY_FRONTIER_V1.md](PLASTICITY_FRONTIER_V1.md), and [RELATIONAL_REGIME_PLASTICITY_V0_1.md](RELATIONAL_REGIME_PLASTICITY_V0_1.md).

## Logical Processing Unit

The LPU is a small stateless exact-computation coprocessor. It is **not** a
symbolic world model.

Cortex decides what temporary values or relations are sufficiently articulated
to send to the unit. The LPU computes only inside that temporary request and
returns a deterministic result plus a witness trace.

v0.1 operations are deliberately small:

- equality and inequality;
- Boolean NOT / AND / OR / implication;
- exact comparisons;
- exact rational ADD / SUB / MUL / DIV;
- directed reachability with a witness path.

The unit stores no facts between calls and cannot create permanent predicates.
In particular, a successful calculation is not automatically written into
Cortex memory.

Conceptually:

```text
L : (temporary articulation, operation) -> (exact result, witness)
```

This preserves the architectural asymmetry:

```text
Cortex: fuzzy, revisable, learned world structure
LPU:    exact, local, stateless computation
```

## Non-goals

v0.1 does not attempt unrestricted theorem proving, first-order ontology,
automatic rule invention inside the LPU, natural-language understanding, or
neural model orchestration.

Those capabilities should only be added when experiments show that they improve
Cortex without freezing its plasticity.
