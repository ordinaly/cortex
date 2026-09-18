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


## 12. Campaign results

### Base learned-world-model run

The base campaign used:

- 12 deterministic seeds;
- perceptual noise \(\sigma_P=0.03\);
- 40 behavioral probe outcomes per real entity;
- 10 interaction opportunities per observed pair.

Mean results:

| Quantity | Result |
|---|---:|
| Identity purity | **1.000** |
| Identity fragmentation | **1.000** |
| Semantic separation ratio | **3.610** |
| Future rare-event Hit@1 at 25% interaction experience | **0.526** |
| Future rare-event Hit@1 at 50% | **0.667** |
| Future rare-event Hit@1 at 100% | **0.823** |
| Interaction-marginal-only Hit@1 at 100% | **0.281** |
| Shuffled-semantic control Hit@1 at 100% | **0.224** |
| Oracle-identity learned model Hit@1 | **0.823** |

Chance among the four rare event types is 0.25.

The principal result is therefore a chronological learning curve:

\[
0.526
\rightarrow
0.667
\rightarrow
0.823.
\]

No semantic family coordinate was supplied to the predictor. The semantic
features used by the bridge were estimated from each Cortex entity's stochastic
behavioral-outcome history.

The interaction-only and shuffled-semantic controls remain near chance. This
supports the finite claim that the learned behavioral semantic geometry carries
transferable information needed by the tensor.

### Identity-noise localization

The perception-noise sweep exposed a sharp composition effect.

At \(\sigma_P=0.03\), Cortex identity remained exact:

\[
\text{purity}=1,
\qquad
\text{fragmentation}=1.
\]

At \(\sigma_P=0.12\), mean identity purity fell to approximately \(0.57-0.73\)
and fragmentation rose to approximately \(1.37-1.69\).

At \(\sigma_P=0.25\), purity fell to approximately \(0.33-0.37\) and
fragmentation rose to approximately \(7.3-8.0\).

The learned semantic geometry collapsed at the same time:

| Perceptual noise | Probe count | Semantic separation | Cortex-bound Hit@1 | Oracle-identity Hit@1 |
|---:|---:|---:|---:|---:|
| 0.03 | 5 | 2.300 | 0.700 | 0.700 |
| 0.03 | 20 | 2.986 | 0.613 | 0.613 |
| 0.03 | 80 | 4.757 | 0.813 | 0.813 |
| 0.12 | 5 | 1.393 | 0.363 | 0.700 |
| 0.12 | 20 | 1.542 | 0.413 | 0.613 |
| 0.12 | 80 | 1.408 | 0.338 | 0.813 |
| 0.25 | 5 | 1.053 | 0.250 | 0.700 |
| 0.25 | 20 | 1.063 | 0.250 | 0.613 |
| 0.25 | 80 | 1.056 | 0.338 | 0.813 |

Each sweep cell contains five seeds, so the non-monotonic differences between
5 and 20 probes should not be interpreted as a stable effect. The robust effect
is the gap between the Cortex-bound and oracle-identity columns once perception
destabilizes binding.

The oracle control receives the **same observed behavioral outcomes and
interaction events** but associates them with stable true identities. Its
prediction quality remains at the clean semantic-learning level even when
perceptual noise is high.

This localizes the main tested failure chain as

\[
\boxed{
\text{perceptual noise}
\rightarrow
\text{identity error}
\rightarrow
\text{mixed/fragmented behavioral statistics}
\rightarrow
\text{semantic collapse}
\rightarrow
\text{forecast degradation}.
}
\]

It does not support the alternative explanation that the tensor bridge itself
becomes intrinsically unstable under perceptual noise.

## 13. Updated interpretation

v1.4-R supports, within this finite synthetic world, the following pipeline:

\[
\boxed{
\text{stable identity}
+
\text{behavioral experience}
\Rightarrow
\text{learned semantic geometry}
\Rightarrow
\text{transfer to unseen pair events}.
}
\]

The experiment also identifies a strong architectural dependency:

\[
\boxed{
\text{semantic learning cannot repair badly corrupted entity identity
after the evidence has already been assigned to the wrong symbols}.
}
\]

This suggests the next engineering/scientific target should not be a more
powerful tensor. It should be a more uncertainty-aware articulation boundary:
provisional identity, delayed commitment, robust observation weighting, and
retrospective reconciliation.

A second parallel route is to remove the predefined outcome alphabet and ask
Cortex to infer latent event channels from trajectories themselves.
