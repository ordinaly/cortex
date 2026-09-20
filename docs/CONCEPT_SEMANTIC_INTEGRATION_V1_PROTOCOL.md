# Concept-to-Semantics Integration v1

Status: **frozen protocol / official campaign passed**.

Protocol ID: **concept-semantic-integration-v1**.

## Research question

Can Cortex ground and revise semantic labels **after** discovering its own latent
concepts, without allowing semantic vocabulary to influence concept formation
itself?

The required direction is:

```math
\text{predictive evidence}
\rightarrow
\text{latent concept } Z
\rightarrow
\text{semantic evidence}
\rightarrow
\text{label belief}.
```

The reverse direction is forbidden in this protocol:

```math
\text{label}
\nrightarrow
\text{concept identity}.
```

## Architectural invariant

A semantic label may depend on a latent concept.

A latent concept must not depend on its semantic label.

This gives three distinct layers:

```math
\text{entity}
\rightarrow
\text{latent concept}
\rightarrow
\text{semantic label}.
```

Concept split, merge, or retraction may invalidate semantic beliefs even when
the old label token remains available.

## Semantic vocabulary surplus

The previous semantic-grounding prototype used equal concept and label counts.

This integration generalizes the finite mapping space to an **injective**
concept-to-label assignment when the vocabulary contains more labels than
active concepts.

For $m$ active concepts and $n\ge m$ candidate labels, the exact mapping space
contains:

```math
P(n,m)
=
\frac{n!}{(n-m)!}
```

assignments.

This permits unused vocabulary and makes it possible for a discovered concept
to remain semantically unresolved rather than forcing every vocabulary item
into use.

## Entity-to-concept evidence projection

Latent concept induction receives only entity/outcome predictive counts.

Only **after** concepts are induced:

- entity-level neural label distributions are projected onto the active concept
  IDs;
- entity-level relational observations are projected onto pairs of active
  concept IDs;
- observations internal to the same discovered concept are not used as
  inter-concept semantic evidence.

The semantic grounder therefore never receives hidden concept-generator labels.

## Experiment A — adversarial semantics after discovery

Six opaque entities contain three predictive concepts of two entities each.

Cortex must first recover the three concepts from predictive counts.

The semantic world contains three true labels plus two unused distractor
labels.

Perception then confidently swaps the labels of two discovered concepts while
leaving the third correct.

Entity-level relational structure remains consistent with the true semantic
mapping.

Required behavior:

1. concept partition F1 remains **1.0**;
2. raw neural semantic accuracy is low;
3. Cortex structural grounding corrects the swapped labels;
4. unused distractor labels are not spuriously resolved.

## Experiment B — label causal isolation

Run concept induction twice on the **same predictive evidence** with completely
different semantic vocabularies and semantic graphs.

Required behavior:

- discovered member blocks are identical;
- persistent concept IDs are identical;
- concept partition F1 remains **1.0**.

This directly checks that semantic vocabulary is downstream of ontology
formation.

## Experiment C — resolved but unnamed concept

Three predictive concepts are discovered.

Two concepts have identifiable semantic labels **A** and **B**.

The third is structurally isolated and receives perceptual support split
symmetrically across three unused vocabulary items.

Required behavior:

- all three predictive concepts are recovered;
- the two nameable concepts resolve correctly;
- the third concept remains semantically unresolved;
- its semantic confidence remains below the acceptance region.

Thus:

```math
Z_{\mathrm{new}}\text{ exists}
```

can coexist with

```math
\operatorname{label}(Z_{\mathrm{new}})
=
\texttt{unresolved}.
```

## Experiment D — semantic lineage under ontology change

A deterministic concept lifecycle runs through:

```math
\text{coarse}
\rightarrow
\text{split}
\rightarrow
\text{merge}
\rightarrow
\text{retract}.
```

Semantic evidence changes consistently with the ontology:

- coarse concept: label **broad**;
- split children: labels **left** and **right**;
- merged concept: label **broad**;
- retracted concept: no active semantic record.

Required behavior:

### Split

The source concept and its semantic label retire.

The two child concepts receive new concept IDs and can ground to **left** and
**right**.

### Merge

The child concept IDs retire.

The merged concept receives a new ID and can ground to **broad**.

### Retraction

The final non-singleton concept disappears.

Its semantic state must disappear with it.

The system must not retain an orphaned semantic belief whose concept no longer
exists.

## Development and official separation

Development seeds: **0..29**.

Official seeds: **800..819**.

The protocol and gates were frozen before this range was evaluated.

## Frozen official gates

The development campaign passed the following gates. They are now frozen unchanged for the untouched official campaign.

### Ontology recovery

1. minimum concept partition F1 in the adversarial fixture = **1.0**;
2. minimum concept partition F1 in the unnamed fixture = **1.0**.

### Semantic correction

3. mean effective Cortex semantic accuracy >= **0.95**;
4. resolved semantic accuracy >= **0.99**;
5. semantic coverage >= **0.95**;
6. Cortex effective semantic accuracy exceeds raw neural accuracy by at least
   **0.50**;
7. distractor-label resolution rate = **0**.

### Causal isolation

8. concept membership and concept IDs remain identical under complete semantic
   vocabulary replacement on every seed.

### Unnamed concepts

9. nameable-concept semantic accuracy = **1.0**;
10. unnamed-concept unresolved rate = **1.0**;
11. unnamed-concept maximum semantic confidence <= **0.50**.

### Semantic lineage

12. split semantic-lineage correctness = **1.0**;
13. merge semantic-lineage correctness = **1.0**;
14. concept retraction invalidates semantic state in every case;
15. split/merge concept identities obey lineage and are never silently reused.

The passing 120-case development campaign measured:

- minimum concept partition F1: **1.000**;
- raw neural semantic accuracy: **0.3333**;
- Cortex semantic accuracy / coverage: **1.000 / 1.000**;
- effective Cortex semantic accuracy: **1.000**;
- distractor-label resolution rate: **0.000**;
- label-causal-isolation rate: **1.000**;
- nameable-concept semantic accuracy: **1.000**;
- unnamed-concept unresolved rate: **1.000**;
- maximum unnamed semantic confidence: **0.3333**;
- split semantic-lineage correctness: **1.000**;
- merge semantic-lineage correctness: **1.000**;
- retraction invalidation correctness: **1.000**;
- concept identity-lineage correctness: **1.000**.

No gate or mechanism may now be changed before official evaluation. Any change
after opening the official seed range requires a new protocol version and fresh
untouched seeds.

## Claim boundary

A passing official campaign would support the narrow claim that, within the
declared finite fixtures, Cortex can:

- discover useful predictive concepts before naming them;
- ground semantics onto those discovered concepts;
- structurally correct adversarial perceptual labels;
- leave a real concept unnamed when vocabulary evidence is insufficient;
- preserve causal independence of concept identity from label vocabulary;
- propagate semantic validity through concept split, merge, and retraction.

It would not establish open-ended language learning, unrestricted semantic
grounding, scalable ontology induction, or real-world multimodal understanding.


## Official outcome

The untouched 80-case official campaign on seeds 800–819 passed every frozen
gate.

Official aggregate metrics:

- minimum latent concept partition F1: **1.000**;
- raw neural semantic accuracy: **0.3333**;
- Cortex semantic accuracy / coverage: **1.000 / 1.000**;
- effective Cortex semantic accuracy: **1.000**;
- distractor-label resolution rate: **0.000**;
- label-causal-isolation rate: **1.000**;
- nameable-concept semantic accuracy: **1.000**;
- unnamed-concept unresolved rate: **1.000**;
- maximum unnamed semantic confidence: **0.3333**;
- split semantic-lineage correctness: **1.000**;
- merge semantic-lineage correctness: **1.000**;
- retraction invalidation correctness: **1.000**;
- concept identity-lineage correctness: **1.000**.

See [CONCEPT_SEMANTIC_INTEGRATION_V1.md](CONCEPT_SEMANTIC_INTEGRATION_V1.md)
and
[the committed aggregate summary](../benchmarks/results/concept_semantic_integration_v1_summary.json).

The frozen gates were not weakened after official evaluation.
