# Semantic revision v1 — official result

Status: **official campaign passed**.

Protocol: [SEMANTIC_REVISION_V1_PROTOCOL.md](SEMANTIC_REVISION_V1_PROTOCOL.md)

Aggregate result:
[`benchmarks/results/semantic_revision_v1_summary.json`](../benchmarks/results/semantic_revision_v1_summary.json)

## Question

Can Cortex preserve a semantic label when contradiction is transient, but revise
that label when contradictory neural and structural evidence persists, without
changing unrelated concepts or flipping directly from one confident label to
another?

## Frozen official campaign

Development seeds `0..19` were used for preflight. After the protocol and
gates were frozen, untouched official seeds `400..419` were evaluated across
three asymmetric semantic families for **60 official cases**.

The official campaign passed every frozen gate.

## Main results

- initial accepted mapping correctness: **1.000**;
- single contradictory neural-shock preservation: **1.000**;
- sustained shifted-mapping revision success: **1.000**;
- original-mapping recovery success: **1.000**;
- required challenged → withdrawn → revised path validity: **1.000**;
- unaffected-concept preservation: **1.000**;
- maximum shift latency: **4 epochs**;
- maximum recovery latency: **4 epochs**;
- bounded neural-history compliance: **1.000**;
- bounded structural-history compliance: **1.000**.

## Interpretation

The accepted semantic label and the instantaneous grounding posterior are now
separate states.

A transient contradiction can change the posterior without immediately
rewriting Cortex's accepted semantic model.

Persistent disagreement follows:

$$
\text{stable}
\rightarrow
\text{challenged}
\rightarrow
\text{unresolved}
\rightarrow
\text{revised stable}.
$$

Thus Cortex does not jump directly between competing confident labels.

Old evidence fades inside a finite horizon. This avoids the opposite failure
mode in which an early label becomes effectively impossible to revise because
historical evidence grows without bound.

The test also verifies locality: only the two concepts whose semantic
correspondence changes are revised. The other three concepts preserve their
accepted labels throughout the shift and recovery sequence.

## Revision memory

v1 keeps at most 12 epochs of evidence and applies age-dependent reliability:

$$
w_N(a)=0.72^a,
\qquad
w_R(a)=0.82^a.
$$

These constants are protocol choices, not claims of optimal forgetting.
Structural evidence decays more slowly than instantaneous neural evidence in
this first prototype.

## Provenance

Every semantic transition records an immutable event carrying:

- epoch;
- concept identity;
- transition kind;
- incumbent label;
- candidate/replacement label;
- posterior confidence;
- posterior margin;
- reason.

The event vocabulary includes:

- `accepted`;
- `challenged`;
- `challenge_cleared`;
- `withdrawn`;
- `revised`;
- `recovered`.

This provides the basis for later proof/dependency tracking of semantic changes.

## Claim boundary

The result supports temporal semantic revision in the declared finite fixtures.
It does not establish optimal belief revision, real-world concept drift,
open-vocabulary semantic change, or trainable feedback into the neural encoder.

The next useful extension is to combine this revision state with noisy
confidence calibration and eventually with concepts produced by latent concept
induction rather than supplied concept identifiers.
