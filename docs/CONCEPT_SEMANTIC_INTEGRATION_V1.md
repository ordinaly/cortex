# Concept-to-Semantics Integration v1 — official result

Status: **official campaign passed**.

Protocol:
[CONCEPT_SEMANTIC_INTEGRATION_V1_PROTOCOL.md](CONCEPT_SEMANTIC_INTEGRATION_V1_PROTOCOL.md)

Aggregate result:
[\`benchmarks/results/concept_semantic_integration_v1_summary.json\`](../benchmarks/results/concept_semantic_integration_v1_summary.json)

## Question

Can Cortex discover its own latent concepts first, then ground semantic labels
onto those concepts without letting semantic vocabulary determine the concept
boundaries?

The official result is **yes within the declared finite fixtures**.

## Frozen official campaign

The mechanism and gates were frozen after a 120-case development campaign.
Untouched official seeds 800–819 were then evaluated across four fixture types,
for **80 official cases**.

Every frozen gate passed.

## End-to-end adversarial naming

Cortex first recovered three latent predictive concepts from opaque
entity/outcome evidence.

Only afterward was semantic evidence introduced.

Perception intentionally swapped the names of two concepts, producing raw
neural semantic accuracy of:

$$
0.3333.
$$

Persistent relational structure then adjudicated the label mapping.

Official result:

- minimum concept partition F1: **1.000**;
- raw neural semantic accuracy: **0.3333**;
- Cortex semantic resolved accuracy: **1.000**;
- Cortex semantic coverage: **1.000**;
- effective Cortex semantic accuracy: **1.000**;
- unused distractor-label resolution rate: **0.000**.

The semantic correction therefore occurred **after** concept discovery. It
cannot be attributed to semantic labels determining the latent partition.

## Label causal isolation

The same predictive evidence was processed twice under completely different
semantic vocabularies and semantic graphs.

Across every official seed:

- discovered concept member blocks were identical;
- persistent concept IDs were identical;
- concept partition recovery remained exact.

Label-causal-isolation rate: **1.000**.

This supports the intended direction:

$$
\text{concept}
\rightarrow
\text{label evidence}
$$

and rejects the reverse dependency inside this integration:

$$
\text{label vocabulary}
\nrightarrow
\text{concept identity}.
$$

## A concept can exist without a name

A separate fixture contained three resolved latent predictive concepts.

Two had identifiable names.

The third was a genuine predictive category but received exactly symmetric
semantic support across three unused vocabulary items.

Official result:

- latent concept partition F1: **1.000**;
- nameable-concept semantic accuracy: **1.000**;
- unnamed-concept unresolved rate: **1.000**;
- maximum unnamed semantic confidence: **0.3333**.

Thus the system can represent:

$$
Z_{\mathrm{new}}\text{ exists}
$$

while simultaneously representing:

$$
\operatorname{label}(Z_{\mathrm{new}})
=
\texttt{unresolved}.
$$

This is a stronger separation between ontology and language than the earlier
semantic-grounding experiment, which received concept identities externally.

## Semantic lineage across ontology change

The final fixture integrated semantic grounding with the latent concept
lifecycle:

$$
\text{coarse}
\rightarrow
\text{split}
\rightarrow
\text{merge}
\rightarrow
\text{retract}.
$$

The semantic trajectory was:

$$
\text{broad}
\rightarrow
\{\text{left},\text{right}\}
\rightarrow
\text{broad}
\rightarrow
\varnothing.
$$

Across every official seed:

- split semantic-lineage correctness: **1.000**;
- merge semantic-lineage correctness: **1.000**;
- semantic invalidation after concept retraction: **1.000**;
- concept identity-lineage correctness: **1.000**.

Concept IDs are not silently reused across membership-changing split or merge
operations.

A semantic belief disappears when the concept it refers to is retracted.

## Surplus vocabulary

The semantic grounder now supports more labels than active concepts.

For $m$ concepts and $n\ge m$ labels, exact inference evaluates injective
assignments:

$$
P(n,m)
=
\frac{n!}{(n-m)!}.
$$

This preserves the original equal-cardinality behavior while allowing unused
vocabulary and semantically unnamed concepts.

The previous frozen semantic-grounding preflight remained green after this
generalization.

## Interpretation

The composed research path is now:

$$
\boxed{
\text{entities}
\rightarrow
\text{latent concept}
\rightarrow
\text{semantic grounding}
}
$$

with concept identity and semantic identity explicitly separate.

This enables several epistemic states that were previously impossible to
represent cleanly:

- known entity, unresolved concept;
- resolved concept, unresolved name;
- resolved concept, corrected name;
- old name invalidated because its concept split;
- multiple child names after concept refinement;
- no semantic belief after concept retraction.

## Claim boundary

This result supports concept-to-semantics composition in the declared bounded
synthetic fixtures.

It does not establish:

- open-vocabulary naming;
- unrestricted semantic taxonomy induction;
- synonym/polysemy resolution;
- semantic inheritance in deep hierarchies;
- natural-language meaning;
- real-world multimodal grounding;
- automatic training feedback into a neural encoder.

The next natural experiment is **hierarchical semantic grounding and semantic
inheritance**, followed by integration with temporal semantic revision so that
labels can persist or change across concept-lineage events without becoming
detached from the ontology they describe.
