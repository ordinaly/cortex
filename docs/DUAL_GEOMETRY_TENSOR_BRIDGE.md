# Cortex v1.3-R — Dual Geometry and Tensor Bridge

Status: **research model / not a software release**.

This version extends v1.1-R semantic geometry and v1.2-R prospective relation
inference by separating two different notions of structure.

## 1. Two geometries

### Semantic geometry

For persistent symbols \(i,j\),

\[
d^S_\rho(i,j)
\]

measures semantic/contextual proximity at semantic resolution \(\rho\).

This answers:

> How similar are the meanings or structural roles of these symbols at this
> semantic scale?

The v1.1-R heat-diffusion construction remains the reference realization.

### Interaction geometry

For entity pairs \((i,j)\) and \((k,m)\),

\[
d^I_\eta((i,j),(k,m))
\]

measures similarity of their **typed interaction profiles** at interaction
specificity \(\eta\).

This answers:

> Do these pairs interact in similar ways at this rarity/specificity scale?

The two metrics are intentionally not collapsed into one scalar.

## 2. Interaction specificity as rarity resolution

Let relation type \(r\) have global empirical frequency \(\pi_r\). Define its
surprisal

\[
s_r=-\log(\pi_r+\epsilon).
\]

Frequent relations have small \(s_r\); rare/specific relations have large
\(s_r\).

For interaction resolution \(\eta\), define a rarity-band kernel

\[
g_\eta(r)
=
\exp\left[
-\frac{(s_r-\eta)^2}{2\sigma_I^2}
\right].
\]

Thus:

- small \(\eta\): common/frequent relations dominate;
- large \(\eta\): rare/specific relations dominate.

For a pair \((i,j)\), let \(\hat p_{ijr}\) be its posterior interaction rate for
relation type \(r\). Define the resolution-conditioned interaction feature

\[
\psi_\eta(i,j)_r
=
\sqrt{
g_\eta(r)\hat p_{ijr}
}.
\]

The interaction metric is

\[
\boxed{
d^I_\eta((i,j),(k,m))
=
\frac1{\sqrt2}
\left\|
\psi_\eta(i,j)-\psi_\eta(k,m)
\right\|_2.
}
\]

This is a Hellinger-style metric on non-normalized interaction intensities.
Unlike a normalized relation distribution, it preserves the fact that one pair
may exhibit a rare interaction much more strongly than another.

## 3. Why the axes are different

Semantic resolution and interaction specificity are independent.

A pair may be:

- semantically close but rarely interact;
- semantically distant but frequently interact;
- semantically distant at fine scale but close at coarse semantic scale;
- interactionally similar for common relations but very different for rare
  relations.

Therefore Cortex should reason in the product space

\[
\boxed{
(\rho,\eta)
}
\]

rather than searching for one universal resolution scalar.

## 4. The interaction tensor

Historical typed events form a three-way tensor

\[
X_{ijr},
\]

where \(i\) is source entity, \(j\) is target entity, and \(r\) is relation/event
type.

This representation is standard in multi-relational learning; three-way tensor
models such as RESCAL are an important neighboring idea. Cortex's research
question is different: how to couple such typed interaction structure to an
online multiresolution semantic geometry without making neural/perceptual
distance itself the semantic metric.

## 5. Tensor bridge

Let

\[
s_i^\rho=\Phi_\rho(i)\in\mathbb R^{d_S}
\]

be the semantic feature vector for symbol \(i\).

Introduce a learned bridge tensor

\[
\boxed{
\mathcal B\in
\mathbb R^{d_S\times d_S\times R}.
}
\]

Each relation slice

\[
B_r=\mathcal B[:,:,r]
\]

maps a pair of semantic states to a predicted interaction intensity:

\[
\tilde p^B_{ijr}
=
f\left[
(s_i^\rho)^\top
B_r
s_j^\rho
\right],
\]

where \(f\) is a non-negative link function.

The tensor therefore learns statements of the form:

> semantic configuration \(A\) interacting with semantic configuration \(B\)
> tends to produce relation/event type \(r\).

This is the bridge between the two geometries.

## 6. Evidence-sensitive fusion

Direct interaction history should dominate when it is well supported, while the
tensor bridge should matter most for sparse or unseen pairs.

Let

\[
c_{ijr}\in[0,1]
\]

be confidence in direct pair/relation evidence.

Define

\[
\lambda_{ijr}^{\rho,\eta}
=
g_\eta(r)
\left[
c_{ijr}\hat p_{ijr}
+
(1-c_{ijr})\tilde p^B_{ijr}
\right].
\]

Then future event probabilities can be obtained by normalizing candidate
intensities:

\[
P(i,r,j\mid \mathcal H_t,\rho,\eta)
=
\frac{
\lambda_{ijr}^{\rho,\eta}
}{
\sum_{a,q,b}
\lambda_{aqb}^{\rho,\eta}
}.
\]

This gives a clean epistemic interpretation:

- **well-known pair**: predict mostly from its own history;
- **poorly observed pair**: borrow structure through semantic compatibility;
- **interaction resolution**: select common versus rare relation regimes.

## 7. Coarse versus fine interaction prediction

Suppose two objects are often merely *near* each other but very rarely engage
in a distinctive *unlock* relation.

At low \(\eta\), the common relation may dominate:

\[
\arg\max_r
\lambda_{ijr}^{\rho,\eta_{\rm coarse}}
=
\text{near}.
\]

At high \(\eta\), the rarity filter can expose:

\[
\arg\max_r
\lambda_{ijr}^{\rho,\eta_{\rm fine}}
=
\text{unlock}.
\]

The model therefore does not need to discard common interactions to represent
rare ones. They occupy different interaction resolutions.

## 8. Event prediction as tensor contraction

The bridge operation is explicitly tensorial:

\[
\boxed{
\tilde{\mathbf p}_{ij}
=
\mathcal B
\times_1 s_i^\rho
\times_2 s_j^\rho.
}
\]

It returns a vector over relation/event types.

The interaction-resolution operator then acts on that relation axis:

\[
\tilde{\mathbf p}_{ij}^{(\eta)}
=
\mathbf g_\eta
\odot
\tilde{\mathbf p}_{ij}.
\]

So semantic resolution acts on the **entity meaning axes**, while interaction
resolution acts on the **relation/event axis**.

This separation is the defining architectural property of v1.3-R.

## 9. First exact reference implementation

The research prototype uses:

1. Hellinger-style semantic features supplied by v1.1-R;
2. empirical typed interaction rates;
3. Gaussian bands in relation surprisal;
4. ridge regression on the Kronecker semantic pair feature
   \(s_i^\rho\otimes s_j^\rho\) to estimate \(\mathcal B\).

Ridge regression is only an exact, auditable small-model reference. It is not a
claim that the final Cortex tensor must be trained by batch least squares.

## 10. Falsifiable toy tests

The first prototype must demonstrate:

1. **interaction-scale reversal** — two pair profiles are close at the common
   relation scale but separate at the rare relation scale;
2. **held-out relation generalization** — a pair with zero direct history gets
   a correct rare event prediction from semantic structure learned on other
   pairs;
3. **resolution-dependent event type** — the same semantic pair can rank a
   common relation at coarse interaction resolution and a rare relation at fine
   resolution;
4. **no semantic collapse** — changing \(\eta\) cannot alter symbol identity or
   \(d^S_\rho\);
5. **no direct leakage** — the held-out pair is absent from tensor fitting.

## 11. Complexity warning

A dense bridge tensor has

\[
O(d_S^2R)
\]

parameters.

That becomes unacceptable for large semantic dimensions or many relation
types. A production model would likely require low-rank factorization, sparse
relation slices, or a structured decomposition.

This is not merely an optimization detail: an unconstrained tensor can also
overfit and destroy the intended interpretability of the two geometries.

The dense reference is therefore restricted to small research fixtures.

## 12. Scientific interpretation

If the tensor bridge predicts unseen typed interactions, the result would
support:

> Semantic structure learned from one set of entities can transfer predictive
> information into the interaction geometry of another pair.

That is stronger than ordinary pair-frequency forecasting but weaker than a
claim of general causal reasoning.

A later causal version would replace relation/event frequencies with
intervention-sensitive outcome channels.

## 13. Research neighborhood

Multi-relational tensor factorization already provides strong precedent for
using a three-way tensor to model entity-relation-entity structure. Temporal
knowledge-graph forecasting likewise uses typed historical relations to predict
future links.

The Cortex-specific research contribution must therefore be evaluated around
the **combination** of:

- independently learned semantic and interaction geometries;
- explicit semantic and rarity resolution axes;
- confidence-gated transfer between direct evidence and tensor prediction;
- online bounded structural learning.

Novelty should not be claimed until a dedicated literature audit is complete.
