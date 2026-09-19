# Cortex Reasoning Map v1 — Structural Analogy / Isomorphism v1

Status: **passed official benchmark / research-only**.

Protocol: `structural-analogy-v1`.

Official run: GitHub Actions run `35436422688`.

## Question

Can Cortex recognize and exploit the same typed relational structure when both
entity labels and relation labels are independently replaced?

The benchmark gives Cortex:

- a complete source typed graph;
- a partially observed target typed graph;
- no shared entity names;
- no shared relation names.

Cortex maintains every joint entity/relation bijection still consistent with
target evidence.

## Core epistemic rule

Let

$$
\mathcal V_t
$$

be the surviving correspondence version space.

For a held-out target structural fact, Cortex resolves only when every mapping
in $\mathcal V_t$ agrees.

This means:

$$
\text{mapping ambiguity}
\neq
\text{predictive ambiguity}.
$$

Several correspondences may remain valid while preserving the same structural
fact.

## Official campaign

Untouched seeds:

$$
300,\ldots,319.
$$

Four graph families and three evidence budgets produced:

$$
20\times4\times3=240
$$

compatible transfer cases.

The full artifact contains 440 rows once identifiability and negative-transfer
diagnostics are included.

Target evidence budgets were:

$$
12,\quad24,\quad36
$$

out of 90 possible directed typed triples.

## Result

Every frozen gate passed.

### Held-out prediction

| observed target facts | mean held-out coverage | resolved accuracy |
|---:|---:|---:|
| 12 / 90 | **0.9718** | **1.000** |
| 24 / 90 | **1.0000** | **1.000** |
| 36 / 90 | **1.0000** | **1.000** |

Positive and negative resolved accuracy were both **1.000**.

The target-only direct-memory baseline had held-out coverage **0.000**.

## No label leakage

Across all cases:

- source/target entity-label overlap: **0**;
- source/target relation-label overlap: **0**.

Therefore transfer could not rely on shared names.

## Mapping identifiability

Under complete target evidence:

- every asymmetric world admitted exactly one joint mapping;
- every symmetric world retained more than one valid mapping;
- every resolved asymmetric entity/relation mapping matched the hidden
  correspondence.

The symmetric fixtures retained exactly the intended automorphism ambiguity.

This matters because Cortex did not manufacture arbitrary certainty merely to
complete the target graph.

## Negative transfer

Two safety controls both passed at **1.000**.

### Near-isomorphic control

A single typed structural change was introduced into asymmetric targets.

With complete diagnostic evidence, Cortex produced an empty correspondence
version space in every case.

### Broken-analogy update

Cortex first formed a useful partial analogy.

Then one explicit target fact was supplied that contradicted every remaining
source correspondence.

Every case transitioned to **incompatible** rather than preserving the old
analogy.

## Candidate version-space behavior

Because target evidence is nested, the correspondence version space should only
shrink.

Measured monotonicity:

$$
1.000.
$$

No compatible case lost the true mapping.

## Interpretation

Within these finite fixtures, Cortex demonstrated a structural analogy
primitive with three useful properties:

1. **surface invariance** — entity and relation labels may both change;
2. **epistemic plurality** — several mappings may remain active when structure
   does not identify one;
3. **conservative transfer** — only consequences shared by every surviving
   mapping are resolved.

The particularly interesting behavior is:

$$
\text{uncertain identity}
+
\text{certain invariant consequence}.
$$

That is stronger than simply running an exact graph-isomorphism test and
selecting one arbitrary mapping.

## Why coverage is so high at low evidence

The 12-fact condition already reached about 97.2% held-out coverage.

This should not be interpreted as a universal sample-complexity result.

The worlds are deliberately small:

- 6 entities;
- 3 relation types;
- exact typed facts;
- a complete source structure;
- finite bijective correspondence;
- deterministic positive/negative target evidence.

These constraints make each observation highly informative.

The next analogy generation should increase graph size, allow partial
correspondence, noisy facts and non-bijective structure before drawing broader
claims.

## Claim boundary

A defensible claim is:

> Within declared finite typed-graph fixtures, Cortex maintained a version
> space of structural correspondences across independently renamed worlds,
> exploited unanimous invariants for held-out prediction, preserved genuine
> automorphism ambiguity, and rejected analogies after explicit structural
> contradiction.

This does not establish unrestricted semantic or human-like analogy.

## Reproducibility

Implementation:

- `research/structural_analogy.py`
- `research/structural_analogy_campaign.py`
- `research/structural_analogy_gate.py`
- `tests/test_structural_analogy.py`

Frozen protocol:

- [`STRUCTURAL_ANALOGY_V1_PROTOCOL.md`](STRUCTURAL_ANALOGY_V1_PROTOCOL.md)

Aggregate result:

- [`benchmarks/results/structural_analogy_v1_summary.json`](../benchmarks/results/structural_analogy_v1_summary.json)

## Next Reasoning Map axis

The next planned reasoning milestone is **latent concept induction**: Cortex
should introduce a new grouping only when that added articulation earns its
complexity by improving predictive structure.
