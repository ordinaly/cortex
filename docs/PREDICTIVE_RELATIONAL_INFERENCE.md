# Cortex v1.2-R — Predictive Relational Inference

Status: **research model / not a software release**.

This model extends v1.1-R multiresolution semantic geometry with a separate
object for predicting future relations.

## 1. Semantic distance is not interaction probability

v1.1-R defines a resolution-dependent semantic pseudometric

\[
d_\rho(i,j)
=
\frac{1}{\sqrt2}
\left\|
(e^{-\rho L}\Phi_0)_i
-
(e^{-\rho L}\Phi_0)_j
\right\|_2.
\]

That answers:

> At this resolution, how similar are the contextual meanings of two symbols?

It should **not** be interpreted directly as

> How likely are the two symbols to interact next?

Two complementary symbols may be semantically close at coarse resolution while
still having a very specific directional or typed interaction pattern.

## 2. Learned relation evidence

For each candidate pair \((i,j)\), Cortex maintains online evidence

\[
s_{ij}=\text{positive interactions},
\qquad
n_{ij}=\text{exposures}.
\]

With a Beta prior \(\mathrm{Beta}(\alpha_0,\beta_0)\),

\[
\hat p_{ij}
=
\frac{\alpha_0+s_{ij}}
{\alpha_0+\beta_0+n_{ij}}.
\]

A confidence factor

\[
c_{ij}
=
\frac{n_{ij}}{n_{ij}+\tau}
\]

prevents one observation from immediately becoming a strong semantic edge.

The first reference affinity is

\[
A_{ij}
=
c_{ij}
\max(0,\hat p_{ij}-p_{\mathrm{bg}}),
\]

where \(p_{\mathrm{bg}}\) is a background interaction rate.

This construction is only a research surrogate. Production Cortex should
eventually support typed relation channels and uncertainty propagation.

## 3. Relational potential

Given

\[
L=D-A,
\]

define the diffusion kernel

\[
K_\rho
=
e^{-\rho L}.
\]

The **relational potential**

\[
\boxed{
\kappa_\rho(i,j)
=
[K_\rho]_{ij}
}
\]

measures how much relation evidence can diffuse from symbol \(i\) to symbol
\(j\) at scale \(\rho\).

Unlike semantic distance, \(\kappa_\rho\) is directly tied to graph connectivity.
For disconnected components,

\[
\kappa_\rho(i,j)=0.
\]

For a pair with no direct edge but with indirect relational paths,
\(\kappa_\rho(i,j)\) can be positive before the pair has ever interacted.

This is the mechanism tested in v1.2-R.

## 4. Deep-path interpretation

The heat kernel expansion is

\[
K_\rho
=
I-\rho L+\frac{\rho^2L^2}{2!}
-\frac{\rho^3L^3}{3!}+\cdots.
\]

If \(i\) and \(j\) have no direct edge and no two-step path, but do have a
three-step path, the first non-zero off-diagonal contribution can appear at
order \(\rho^3\).

A bipartite compositional structure can therefore support a prediction such as

\[
\text{tool}_1
\to
\text{target}_2
\to
\text{tool}_2
\to
\text{target}_1
\]

even when the missing pair
\((\text{tool}_1,\text{target}_1)\) has never been observed.

This gives a precise meaning to **deep relation inference**.

## 5. Prospective prediction contract

At a cutoff time \(T\):

1. build relation evidence using observations strictly before \(T\);
2. freeze the graph for scoring;
3. score pairs that have **never been directly exposed**;
4. reveal which of those pairs interact in the future;
5. evaluate the pre-event ranking.

Ground-truth domain labels are never input to the learner.

The primary metrics are:

- AUROC over unseen candidate pairs;
- average precision;
- mean reciprocal rank of future positive pairs;
- Hit@1 and Hit@3.

## 6. Baselines

The first campaign compares:

1. **Direct posterior only** — unseen pairs all have the same prior score.
2. **Fine informational distance** — uses \(d_0\) only.
3. **Common neighbors** — a two-hop graph baseline.
4. **Three-hop path score** — \([A^3]_{ij}\).
5. **Cortex relational potential** — \(\kappa_\rho(i,j)\).

The three-hop baseline is important: the synthetic world is deliberately built
so that the missing relation is supported first by three-step paths. The
diffusion kernel therefore should not be credited merely for discovering that
multi-hop paths exist; its stronger claim is that one resolution operator sums
evidence from all path depths.

## 7. Synthetic compositional world

The first world contains several hidden interaction domains. Each domain has:

- three actor-like entities;
- three receiver-like entities.

Actors and receivers have the same fine informational roles across domains, so
\(d_0\) cannot reveal which actor belongs with which receiver.

Within-domain actor/receiver interactions are frequent. Cross-domain
interactions exist at a lower noisy background rate.

One valid within-domain pair per domain is completely withheld during training.
Several cross-domain pairs are also withheld. The learner therefore sees no
direct evidence for any evaluated pair.

The task is to rank which unseen pair will become a future interaction.

## 8. Interpretation

A successful finite experiment supports only this claim:

> In the tested structured worlds, relation evidence learned from past
> interactions can create a diffusion geometry that ranks previously unseen
> future interactions above unseen distractors.

It does not establish universal relational reasoning or real-world prediction.

## 9. Complexity warning

The exact reference uses dense matrix exponentials and dense matrix powers.
That is approximately cubic time and quadratic memory in the number of symbols.

It must not be moved into the Cortex online hot path as-is.

The production candidate remains a sparse approximation such as Krylov action,
Chebyshev expansion, or bounded local diffusion, validated against the exact
reference on small graphs.

## 10. Next step after the synthetic gate

If the prospective synthetic test succeeds, repeat the protocol on persistent
entities produced by the neural/Cortex perception stack:

\[
\text{pixels}
\to
\text{neural symbols}
\to
\text{Cortex entities}
\to
\text{learned relations}
\to
\text{future interaction prediction}.
\]

That is the first experiment where Cortex would be asked to reason prospectively
over what a neural network sees rather than only maintaining identity.
