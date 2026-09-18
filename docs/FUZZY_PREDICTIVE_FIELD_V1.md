# Cortex v1.6-R — Fuzzy Predictive Field

Status: **research model / not a production release**.

v1.5-R showed that delaying identity commitment repairs part of the downstream
reasoning damage caused by hard binding errors. v1.6-R asks a more fundamental
question:

> Does Cortex need discrete persistent identity internally at all?

The new model treats exact physical identity as an optional projection supplied
by perception when needed. Cortex itself maintains a fuzzy field of predictive
possibilities.

## 1. Perceptual anchors are not ontology

The neural/perceptual layer may expose a finite set of track or memory anchors

\[
A=\{a_1,\ldots,a_m\}.
\]

These anchors are support points for evidence. Cortex does **not** interpret them
as a claim that the world contains exactly \(m\) persistent entities.

For observation \(o_t\), perception provides a possibility profile

\[
\Pi_t^P(a)
=
\exp\left[
-\frac{d_P(o_t,a)}{\tau_P}
\right]
\in(0,1].
\]

The profile is not required to sum to one.

When finite evidence must be allocated, Cortex derives responsibilities

\[
w_t(a)
=
\frac{\Pi_t^P(a)}
{\sum_b\Pi_t^P(b)}.
\]

The normalization is a computational evidence-allocation step, not an
ontological assertion.

## 2. Fractional evidence ownership

If outcome \(y_t\) is observed, semantic evidence is updated fractionally:

\[
N_a(y_t)
\leftarrow
N_a(y_t)+w_t(a).
\]

For a pair of observations \(o_t^A,o_t^B\) participating in event \(r_t\),

\[
N_{ab}(r_t)
\leftarrow
N_{ab}(r_t)
+
w_t^A(a)w_t^B(b).
\]

A single uncertain correspondence can therefore no longer poison one discrete
entity history.

## 3. Predictive semantic field

Each anchor obtains an informational feature

\[
\phi(a)
=
\sqrt{
\frac{N_a+\alpha}
{\|N_a\|_1+\alpha K}
}.
\]

Define predictive-semantic distance

\[
d^P(a,b)
=
\frac1{\sqrt2}
\|\phi(a)-\phi(b)\|_2.
\]

At field resolution \(\rho\), define the fuzzy affinity kernel

\[
\boxed{
K_\rho(a,b)
=
\exp\left[
-\frac{d^P(a,b)^2}{2\rho^2}
\right].
}
\]

Small \(\rho\) preserves predictive distinctions. Large \(\rho\) allows
predictively similar anchors to support one another.

For a new observation, its Cortex field activation is

\[
\boxed{
f_t^\rho
=
\operatorname{Normalize}
\left[
w_t K_\rho
\right].
}
\]

This field can have support across several physically distinct anchors.

## 4. No requirement for exact identity

Suppose three physical tools are perceptually confusable and have the same
predictive role.

The neural layer may have low exact top-1 correspondence accuracy while placing
almost all possibility mass on the correct predictive family.

Cortex is allowed to maintain

\[
f_t^\rho
\approx
(0.34,0.33,0.33)
\]

across the three anchors if they induce essentially the same future.

That representation is physically inaccurate but predictively sufficient.

The research objective is therefore not world reconstruction fidelity. It is:

\[
\boxed{
\text{minimal internal distinction subject to bounded predictive loss}.
}
\]

This is closely related in spirit to predictive-state representations,
predictive rate-distortion and past-future information-bottleneck ideas. The
v1.6-R combination with the existing Cortex semantic/interaction resolution
machinery should not be claimed as novel without a dedicated literature audit.

## 5. Relational field and tensor contraction

The v1.3-R bridge tensor remains:

\[
\mathcal B\in
\mathbb R^{d_S\times d_S\times R}.
\]

But the tensor no longer receives two definite entity states.

For source and target observations,

\[
\bar\phi_A
=
\sum_a f_A^\rho(a)\phi(a),
\]

\[
\bar\phi_B
=
\sum_b f_B^\rho(b)\phi(b).
\]

Then

\[
\tilde p_r
=
\bar\phi_A^\top B_r\bar\phi_B.
\]

Interaction specificity remains controlled by relation surprisal:

\[
g_\eta(r).
\]

The event field is

\[
\boxed{
\Lambda_r^{\rho,\eta}
=
g_\eta(r)
\,
\bar\phi_A^\top B_r\bar\phi_B.
}
\]

Only the external prediction readout needs normalization across event types.

## 6. Equivalent integral form

The same operation can be written directly over anchor possibilities:

\[
\Lambda_r^{\rho,\eta}
=
g_\eta(r)
\sum_{a,b}
f_A^\rho(a)
f_B^\rho(b)
\phi(a)^\top B_r\phi(b).
\]

Identity is therefore an integration variable, not a prerequisite for
reasoning.

## 7. Predictive complexity

A fuzzy model must not gain predictive accuracy merely by storing every possible
world.

For the semantic kernel eigenvalues \(\lambda_i\), define effective predictive
rank

\[
N_{\rm eff}
=
\exp\left[
-\sum_i
p_i\log p_i
\right],
\qquad
p_i=
\frac{\lambda_i}{\sum_j\lambda_j}.
\]

If 24 perceptual anchors collapse into eight predictive roles, a useful fuzzy
representation should have

\[
N_{\rm eff}\ll24
\]

while preserving future-event accuracy.

This provides a first measurable approximation to predictive-state complexity.

## 8. First falsifiable world

The first campaign deliberately makes exact physical identity difficult but
predictive role easy.

There are:

- 24 physical individuals;
- eight predictive families;
- three individuals per family.

Perceptual anchor centers are nearly identical within a family and strongly
separated across families.

Thus the neural matcher can have poor exact identity top-1 accuracy while still
placing nearly all possibility mass on the correct predictive family.

Behavioral outcome distributions and future typed interactions depend only on
predictive family, not individual identity.

The campaign compares:

1. **hard perceptual projection** — argmax correspondence before all updates;
2. **fuzzy predictive field** — fractional updates and field contraction;
3. **oracle physical identity**.

The decisive result is not necessarily that fuzzy prediction beats hard
prediction. It is that fuzzy prediction remains close to oracle while exact
identity fidelity is substantially lower and effective predictive complexity is
closer to the number of roles than to the number of individuals.

## 9. Adaptive distinction gate

The second fixture tests whether fuzziness is reversible.

Several anchors begin with the same predictive outcome distribution. One anchor
is then given evidence from a different future-outcome distribution.

The model should require no symbolic split operation. Instead,

\[
d^P(a,b)
\]

increases continuously and

\[
K_\rho(a,b)
\]

contracts.

Thus a distinction emerges automatically when it becomes predictively useful.

The desired principle is:

\[
\boxed{
\text{prediction creates distinctions;
ontology does not create prediction}.
}
\]

## 10. Scientific pass conditions

The initial campaign should show:

1. exact neural top-1 identity is materially below 1;
2. correct-family possibility mass remains high;
3. fuzzy future-event accuracy is substantially above chance and close to
   oracle;
4. shuffled or semantically destroyed field structure degrades prediction;
5. effective predictive rank is substantially below physical anchor count;
6. when one predictive role changes, its field affinity to the former role
   decreases without any hard identity split.

## 11. Non-claims

The first v1.6-R prototype assumes that the neural system exposes candidate
perceptual anchors and correspondence scores.

It does not yet solve anchor creation, deletion or neural tracking.

It also uses the existing dense research tensor and predefined behavioral
outcome channels.

The experiment tests only whether a fuzzy predictive representation can make
hard internal identity unnecessary for the downstream reasoning task.


## 12. Base campaign result

The first v1.6-R campaign completed successfully on 20 deterministic seeds.

Mean base results:

| Quantity | Result |
|---|---:|
| Exact physical identity top-1 | 0.480 |
| Correct predictive-family possibility mass | 0.9998 |
| Hard-projection future Hit@1 | 0.706 |
| **Fuzzy-field future Hit@1** | **0.909** |
| Oracle hard-identity Hit@1 | 0.622 |
| Oracle identity + predictive field Hit@1 | 0.781 |
| Effective predictive rank | 8.64 |
| Physical anchors | 24 |

The synthetic world contained eight predictive roles and three physical
individuals per role.

Thus the field retained substantially less effective predictive complexity than
the physical anchor count while achieving better prospective accuracy than the
hard correspondence projection.

The correct interpretation is not that inaccurate knowledge is intrinsically
superior to accurate identity. In this sparse fixture, physically distinct
individuals have equivalent futures, so hard identity divides evidence across
distinctions that do not help prediction.

The result supports:

\[
\boxed{
\text{predictive sufficiency can survive poor exact identity fidelity}.
}
\]

### Perceptual noise sweep

| Perceptual noise | Exact identity | Correct-family mass | Fuzzy Hit@1 | Hard Hit@1 |
|---:|---:|---:|---:|---:|
| 0.08 | 0.638 | ~1.000 | 0.831 | 0.619 |
| 0.12 | 0.535 | ~1.000 | 0.831 | 0.631 |
| 0.20 | 0.456 | 0.997 | 0.819 | 0.650 |
| 0.25 | 0.423 | 0.977 | 0.806 | 0.594 |

The forecast therefore degrades much more slowly than exact physical
correspondence.

## 13. Adaptive predictive distinction campaign

The second v1.6-R campaign tests reversibility of fuzzy equivalence.

Three perceptually similar physical anchors begin with the same future-outcome
distribution. Their predictive field should therefore form one basin.

After the initial phase, one physical anchor develops a different future.
No identity split, concept split or symbolic reclassification is supplied.

The model is evaluated on:

- exact perceptual top-1 identity;
- affinity between the changed and unchanged anchors;
- affinity between the two unchanged anchors;
- effective predictive rank;
- probability assigned to the newly characteristic future outcome;
- adaptation trajectory over time.

The desired behavior is

\[
K_\rho(a_0,a_1)
\downarrow
\]

while

\[
K_\rho(a_1,a_2)
\]

remains high, together with

\[
N_{\rm eff}
\uparrow.
\]

This is a direct falsification of the principle

\[
\boxed{
\text{prediction creates distinctions when they become useful}.
}
\]

Exact physical identity is deliberately imperfect in this fixture. The
experiment therefore asks whether repeated weak correspondence evidence plus
divergent futures are sufficient for the field to sharpen without requiring
discrete ontology.
