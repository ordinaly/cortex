# Cortex v1.4-R — Learned Semantic World Model

Status: **research model / not a software release**.

v1.4-R removes a major privilege from v1.3-R: the tensor bridge no longer
receives hand-authored semantic coordinates.

The system must derive semantic features from chronological experience before
using them to predict held-out future interactions.

## 1. Research question

Can the Cortex stack perform the following chain without semantic family labels
being supplied to the predictor?

\[
\text{noisy perceptual evidence}
\rightarrow
\text{persistent entities}
\rightarrow
\text{behavioral statistics}
\rightarrow
\Phi_0
\rightarrow
\mathcal B
\rightarrow
\text{future typed event}.
\]

The hidden world still contains semantic families so that the experiment has
ground truth, but those family labels are used only by the generator and
evaluation code.

## 2. Perceptual layer

Each real entity has a fixed latent perceptual center

\[
z_i\in\mathbb R^{32}.
\]

The centers are randomly generated orthogonal directions. They identify
individual objects but deliberately contain **no semantic-family structure**.

Each observation is

\[
\tilde z_{i,t}
=
\operatorname{normalize}(z_i+\xi_t),
\qquad
\xi_t\sim\mathcal N(0,\sigma_P^2I).
\]

These vectors stand in for the output of a frozen neural encoder.

They are passed through the existing \`HybridCortexRuntime\`, which performs
persistent entity articulation using the cosine-calibrated metric bridge.

Thus perceptual identity and semantic role are deliberately different sources
of information.

## 3. Semantic learning from experience

There are four source-side and four target-side hidden behavioral families.

Cortex is never given the family integer.

Instead, every entity participates in ordinary probe experiences with observed
outcomes

\[
Y_t\in\{1,\ldots,8\}.
\]

Entities from the same hidden family generate similar outcome distributions,
but each observation is stochastic.

For internal Cortex entity \(e\), maintain Dirichlet-smoothed outcome counts

\[
N_e(y).
\]

The learned fine semantic representation is

\[
\boxed{
\phi_0(e)_y
=
\sqrt{
\frac{N_e(y)+\alpha}
{\sum_q N_e(q)+8\alpha}
}.
}
\]

Therefore semantic distance is learned from predictive/behavioral consequences,
not from perceptual appearance:

\[
d^S_0(e,f)
=
\frac1{\sqrt2}
\|\phi_0(e)-\phi_0(f)\|_2.
\]

This is the first v1.x campaign where the semantic tensor coordinates are
estimated from the stream itself.

## 4. Interaction stream

After semantic probe experience, source and target entities undergo typed
interaction opportunities.

Relations are:

1. \`near\` — common;
2. \`contact\` — medium;
3. four rare typed interactions.

For hidden source family \(a\) and target family \(b\), the rare type is

\[
r(a,b)=2+(a+b)\bmod4.
\]

The exact same Latin-square structure used in the v1.3-R ablation is retained
because it prevents source-only or target-only marginals from solving the task.

One concrete source/target pair for each family combination is held out from
the entire pre-cutoff interaction stream.

## 5. Chronological contract

The experiment has two chronological phases.

### Phase A — semantic acquisition

Entities are observed repeatedly and stochastic behavioral outcomes are
recorded against the **Cortex binding**, not the ground-truth identity.

### Phase B — interaction acquisition

Observed non-held-out entity pairs are presented in random chronological order.
Cortex must bind both observations. Typed event counts are accumulated against
the resulting internal entity pair.

Predictions are frozen at interaction-stream fractions

\[
25\%,\quad50\%,\quad100\%.
\]

At each cutoff, the tensor bridge is fitted only from evidence available before
that cutoff.

## 6. Strict prospective evaluation

For every held-out true pair:

- the pair has never interacted before the cutoff;
- the corresponding internal Cortex pair is removed from bridge fitting even
  if identity aliasing would otherwise create an accidental collision;
- the model predicts the future rare event type;
- only then is ground truth used for scoring.

This distinction is important: identity mistakes are allowed to hurt the model,
but they are not allowed to create direct pair-frequency leakage.

## 7. Measured layers

The campaign records four distinct properties.

### Identity stability

- binding purity;
- average fragmentation;
- number of active internal entities.

### Learned semantic structure

Using hidden families only for evaluation:

- mean within-family semantic distance;
- mean between-family semantic distance;
- separation ratio

\[
S_{\rm sem}
=
\frac{\bar d_{\rm between}}
{\bar d_{\rm within}}.
\]

### Event prediction

Compare:

- semantic-only chance control;
- interaction-marginal-only;
- tensor without rarity resolution;
- learned-semantics tensor + rarity resolution;
- shuffled-learned-semantics control.

### Learning curve

Measure rare-event Hit@1 at 25%, 50% and 100% of interaction experience.

## 8. Hardness axes

Two factors are varied independently:

\[
\sigma_P
=
\text{perceptual embedding noise},
\]

and

\[
n_S
=
\text{semantic probe observations per entity}.
\]

This separates failure due to unstable identity from failure due to poorly
estimated meaning.

## 9. Intended interpretation

A successful result would support:

> Cortex can convert stable perceptual identities into learned behavioral
> semantic coordinates and reuse those coordinates to predict an interaction
> for a pair that never interacted before.

That would be a stronger result than v1.3-R because the semantic coordinates
would no longer be supplied by the experimenter.

It would still not demonstrate that Cortex discovered the behavioral outcome
channels themselves. The outcome alphabet is predefined in v1.4-R.

## 10. Next boundary

If v1.4-R succeeds, v1.5-R should remove another privilege:

\[
\boxed{
\text{predefined outcome/relation channels}.
}
\]

The system would then need to infer latent event types from trajectories and
consequences before constructing the dual geometry.

## 11. Complexity warning

The dense tensor remains the exact small-system research reference and retains
the

\[
O(d_S^2R)
\]

parameter cost from v1.3-R.

The online stream itself does not justify moving this dense fit into the native
hot path. A low-rank or sparse online bridge should be investigated only after
the learned-semantic experiment establishes that the architecture is useful.
