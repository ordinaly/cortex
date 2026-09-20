# Hybrid Cortex semantic grounding v1 — mutual perception/reasoning

Status: **frozen protocol / official campaign passed**.

Protocol ID: `hybrid-semantic-grounding-v1`.

This experiment defines the first inference-time bidirectional semantic boundary
between neural perception and Cortex reasoning.

It does **not** change the stable native runtime, the frozen executable
specification, or Hybrid Stage I. It is a research-only Stage-II-A prototype
that operates on concept identities supplied to the grounding layer.

## Research question

Can neural perception and Cortex reasoning reinforce one another when attaching
human-readable labels to internal concepts, while allowing Cortex to demote a
high-confidence perceptual label when persistent structure contradicts it?

The architectural rule is:

$$
\boxed{
\text{perception proposes}
\rightarrow
\text{Cortex adjudicates}
\rightarrow
\text{perception may be queried again}
}
$$

with the additional epistemic constraint:

$$
\boxed{
\text{later evidence may overturn both layers.}
}
$$

Cortex therefore has the final word only over its **current accepted semantic
state**, not over objective truth.

## Separation of concept identity and label

The most important invariant is:

$$
\boxed{
\text{concept identity must not depend on its current label.}
}
$$

The grounding layer receives persistent concept identifiers

$$
Z_1,\ldots,Z_n
$$

and candidate semantic labels

$$
\ell_1,\ldots,\ell_n.
$$

Changing the accepted label for $Z_i$ never changes the identity of $Z_i$.

This lets Cortex represent a stable concept with an unresolved name, revise a
name after later evidence, or demote a neural proposal without deleting the
underlying concept.

Latent-concept induction remains a separate preceding research problem. v1
isolates semantic grounding by assuming concept identities already exist.

## v1 scope

v1 deliberately uses a finite flat one-to-one vocabulary:

$$
\phi:\mathcal Z\rightarrow\mathcal L
$$

where $|\mathcal Z|=|\mathcal L|$ and $\phi$ is bijective.

This restriction makes the complete mapping posterior auditable.

v1 does **not** yet model synonyms, hierarchical labels, open-vocabulary label
creation, unknown labels outside the supplied vocabulary, gradient feedback
into the neural encoder, or latent-concept discovery itself.

## Neural evidence

For each concept $Z_i$, perception emits a complete candidate distribution

$$
P_N(\ell\mid Z_i).
$$

The neural distribution is evidence, not accepted Cortex state.

Repeated perceptual observations may accumulate with an explicit reliability
weight $r_N>0$. For a proposed mapping $\phi$, neural log-evidence is

$$
S_N(\phi)
=
\sum_i\sum_t
r_{it}
\log
P_{N,t}\bigl(\phi(Z_i)\mid Z_i\bigr).
$$

## Structural semantic evidence

Cortex also observes typed structural facts between its internal concepts:

$$
(Z_i,r,Z_j,\text{present})
$$

or

$$
(Z_i,r,Z_j,\text{absent}).
$$

The external semantic vocabulary supplies a typed semantic relation graph over
candidate labels.

Under mapping $\phi$, a structural observation agrees when

$$
(\phi(Z_i),r,\phi(Z_j))
$$

has the same present/absent value in the semantic graph.

Let $q\in(0.5,1)$ be the declared reliability of a structural match. The
structural score is

$$
S_R(\phi)
=
\sum_k r_k
\begin{cases}
\log q, & \text{if observation }k\text{ agrees},\\
\log(1-q), & \text{otherwise}.
\end{cases}
$$

The current implementation uses $q=0.95$.

## Joint adjudication

The total mapping score is

$$
S(\phi)=w_N S_N(\phi)+w_R S_R(\phi),
$$

with frozen v1 defaults

$$
w_N=w_R=1.
$$

Normalized finite mapping support is

$$
P_C(\phi)
=
\frac{\exp S(\phi)}
{\sum_{\psi}\exp S(\psi)}.
$$

This is exact normalized support over the declared finite hypothesis space. It
must not be described as a universally calibrated probability that a label is
metaphysically correct.

The Cortex label marginal is

$$
P_C(\ell\mid Z_i)
=
\sum_{\phi:\phi(Z_i)=\ell}
P_C(\phi).
$$

## Resolution rule

For concept $Z_i$, let the largest and second-largest label marginals be $p_1$
and $p_2$.

v1 resolves a semantic label only if

$$
p_1\ge 0.70
$$

and

$$
p_1-p_2\ge 0.15.
$$

Otherwise the accepted semantic state is `unresolved`. No arbitrary tie-break
is allowed.

## Cortex-to-perception feedback

The grounding layer emits one of four inference-time feedback actions.

- **`reinforce`** — Cortex resolves the same label as the neural top candidate.
- **`reinspect`** — Cortex resolves a different label; feedback contains the
  promoted and demoted labels plus a revision strength.
- **`broaden`** — structural evidence challenges the neural ranking, but
  Cortex is not confident enough to relabel.
- **`hold`** — evidence remains insufficient for revision.

`reinspect` is the explicit mechanism by which Cortex may demote a neural
label. `broaden` is equally important because disagreement does not imply that
Cortex is already correct.

## Why the loop is not yet gradient feedback

v1 is **Stage II-A**, an inference-time semantic loop. The neural model may
receive a `reinspect` or `broaden` request and produce another candidate
distribution, but Cortex does not modify neural parameters.

This lets us test whether structural feedback is useful and conservative before
adding a trainable feedback path.

A later Stage II-B may test

$$
\text{encoder}
\leftrightarrow
\text{Cortex}
$$

with representation learning, but that requires separate stability and causal
attribution controls.

## Development worlds

Each normal world contains:

- 5 opaque Cortex concept identifiers;
- 5 disjoint opaque candidate labels;
- 2 typed relation kinds;
- one hidden concept-to-label bijection;
- one of three asymmetric structural families.

Concept strings and label strings are disjoint, preventing token-name leakage.
The hidden bijection is used only for evaluation.

## Neural corruption

The clean perceptual condition gives the true label probability 0.84.

Adversarial conditions swap the neural top labels of 0, 1, or 2 concept pairs.
A corrupted concept assigns 0.90 probability to the wrong swapped label and
only 0.08 to the true label.

Because corruption is pairwise, the neural top assignments remain a valid
bijection. A global one-to-one assignment alone therefore cannot repair the
swap; structural evidence is required.

The resulting neural top-label accuracies are exactly:

$$
1.0,\quad 0.6,\quad 0.2.
$$

## Structural evidence budgets

Normal cases use nested structural budgets:

$$
0,\quad 8,\quad 12.
$$

Budget 0 is the neural-only/global-assignment control. Positive and negative
structural facts are interleaved in a deterministic seeded order.

## Symmetry control

A separate symmetric world contains two structurally interchangeable concepts.
Their neural distributions are also exactly symmetric between the two
corresponding labels.

Even with complete structural evidence, Cortex must preserve

$$
P_C(\ell_a\mid Z)=P_C(\ell_b\mid Z)=0.5
$$

for the twin concepts and return `unresolved`. Other non-symmetric concepts
must still resolve correctly.

## Development and official seeds

Development/preflight seeds are $0,\ldots,19$.

Official untouched seeds are reserved as $300,\ldots,319$.

The protocol was frozen before the official seed range was evaluated.

## Development preflight result

The development campaign contains 560 rows.

With one adversarial swapped pair:

- neural top-label accuracy: **0.600**;
- at structural budget 8, Cortex resolved accuracy: **0.9933**;
- coverage: **1.000**;
- correction rate: **0.9833**;
- `reinspect` precision: **0.9833**;
- false reinspection on neural-correct concepts: **0.000**.

With two swapped pairs:

- neural top-label accuracy: **0.200**;
- at structural budget 12, Cortex resolved accuracy: **0.9311**;
- coverage: **0.980**;
- effective correct fraction: **0.920**;
- correction rate: **0.900**;
- false reinspection on neural-correct concepts: **0.000**.

On clean perception, resolved accuracy and coverage are **1.000**, with zero
false reinspection.

On the exact symmetry control, the twin unresolved fraction is **1.000**, mean
twin label support is **0.500**, and non-twin resolved accuracy is **1.000**.

These are development results only. They are not an official untouched claim.

## Frozen gates

The official campaign uses the same implementation, parameters and thresholds
and passes only if every gate below succeeds.

### Integrity

1. protocol identifier is exact;
2. all expected seeds/families/budgets are present;
3. concept/label namespace overlap = **0**.

### Clean preservation

4. clean resolved accuracy = **1.0**;
5. clean coverage = **1.0**;
6. clean false-reinspection rate = **0.0**.

### One swapped pair

7. neural fixture accuracy = **0.6**;
8. Cortex resolved accuracy at budget 8 >= **0.98**;
9. coverage >= **0.95**;
10. correction rate >= **0.95**;
11. `reinspect` precision >= **0.95**;
12. false-reinspection rate = **0.0**.

### Two swapped pairs

13. neural fixture accuracy = **0.2**;
14. Cortex resolved accuracy at budget 12 >= **0.90**;
15. coverage >= **0.90**;
16. effective correct fraction >= **0.85**;
17. correction rate >= **0.85**;
18. false-reinspection rate = **0.0**.

### Ambiguity

19. symmetric twin unresolved fraction = **1.0**;
20. non-twin resolved accuracy = **1.0**;
21. mean symmetric support remains **0.5** within numerical tolerance.

No official threshold may be weakened after official results are observed.

## Negative-result policy

If the untouched official campaign fails, preserve the failure and revise the
protocol only on fresh untouched seeds.

A lower false-demotion threshold must never be justified post-hoc by claiming
that Cortex is "supposed" to overrule perception. The experiment tests whether
Cortex can correct perception **without becoming a second source of unjustified
certainty**.

## Implementation

Research-only files:

- `research/semantic_grounding.py`
- `research/semantic_grounding_campaign.py`
- `research/semantic_grounding_gate.py`
- `tests/test_semantic_grounding.py`

The stable native runtime is unchanged.

## Intended claim boundary

A passing official campaign would support the narrow claim that, within the
declared finite semantic-grounding fixtures, neural candidate labels and Cortex
relational evidence can be combined in an auditable global mapping posterior;
Cortex can demote adversarially incorrect high-confidence neural labels,
preserve correct neural labels, return explicit reinspection feedback, and
retain unresolved state when structural symmetry makes the label correspondence
genuinely ambiguous.

It would not establish open-vocabulary semantic understanding, natural-language
grounding, or superiority over trained multimodal foundation models.


## Official outcome

The untouched official campaign on seeds 300–319 passed every frozen gate.

Key results:

- clean perception: **1.000 resolved accuracy / 1.000 coverage / 0 false reinspection**;
- one swapped pair: neural **0.600** → Cortex **0.9933 resolved accuracy**, **0.9933 coverage**, **0.9667 correction rate** at 8 structural facts;
- two swapped pairs: neural **0.200** → Cortex **0.960 resolved accuracy**, **1.000 coverage**, **0.960 effective correct fraction**, **0.950 correction rate** at 12 facts;
- false reinspection on neural-correct concepts: **0.000** in both corrupted conditions;
- exact symmetric twins: **1.000 unresolved fraction** with mean support **0.500**;
- non-symmetric labels in the symmetry control: **1.000 resolved accuracy**.

See [HYBRID_SEMANTIC_GROUNDING_V1.md](HYBRID_SEMANTIC_GROUNDING_V1.md)
and [the committed aggregate summary](../benchmarks/results/hybrid_semantic_grounding_v1_summary.json).

The frozen thresholds above were not weakened after evaluation.
