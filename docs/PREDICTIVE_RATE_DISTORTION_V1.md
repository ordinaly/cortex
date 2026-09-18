# Cortex v1.7-R — Predictive Rate-Distortion and Adaptive Resolution

Status: **research model / not a production release**.

v1.6-R established two finite results:

1. exact physical identity can be poor while a fuzzy predictive field remains
   accurate;
2. predictive distinctions can emerge continuously when futures diverge and
   become weaker again at coarser resolution.

v1.7-R removes the remaining hand-selected semantic resolution.

## 1. Objective

Cortex should keep the lowest-complexity field that preserves useful future
prediction.

For candidate field resolution \(\rho\), let

\[
D_t(\rho)
\]

be recent prequential predictive distortion and

\[
C_t(\rho)
\]

be current field complexity, measured initially by effective predictive rank.

The reference controller solves

\[
\boxed{
\rho_t^*
=
\arg\min_{\rho}
C_t(\rho)
\quad
\text{subject to}
\quad
D_t(\rho)
\le
D_t^{\min}+\varepsilon.
}
\]

This is a constrained rate-distortion rule rather than a weighted loss with an
arbitrary complexity coefficient.

## 2. Prequential distortion

For each incoming outcome, every candidate resolution predicts **before** the
outcome is observed.

For candidate \(\rho\),

\[
\ell_t(\rho)
=
-\log
P_\rho(y_t\mid o_t,H_{t-1}).
\]

Only after all candidate predictions are frozen is \(y_t\) incorporated into
the field.

This prevents retrospective selection of resolution from the same observation
used to train the field.

## 3. Regional rather than global loss

Average loss can hide a rare region whose future has become different.

The controller therefore maintains fuzzy per-anchor exponentially decayed loss
statistics.

Observation \(o_t\) supplies perceptual responsibilities \(w_t(a)\). For each
candidate resolution,

\[
L_{\rho,a}
\leftarrow
\lambda L_{\rho,a}
+
w_t(a)\ell_t(\rho),
\]

with similarly decayed anchor mass.

The current distortion is the worst supported regional loss:

\[
\boxed{
D_t(\rho)
=
\max_{a\in A_t^{\rm active}}
\frac{L_{\rho,a}}{M_a}.
}
\]

Thus a minority predictive regime can force additional resolution when it
actually becomes necessary.

## 4. Complexity

The first reference uses the v1.6-R effective kernel rank

\[
C_t(\rho)
=
N_{\rm eff}(K_\rho).
\]

This is not claimed to be the final information-theoretic rate.

It is an auditable finite surrogate for how many predictive degrees of freedom
the field currently expresses.

## 5. Adaptive three-phase campaign

The world contains three perceptually ambiguous anchors.

### Phase A — equivalent futures

All three share the same future distribution.

Desired behavior:

\[
\rho\uparrow,
\qquad
C(\rho)\downarrow.
\]

### Phase B — divergent future

One anchor develops a different future.

Desired behavior:

\[
\rho\downarrow,
\qquad
C(\rho)\uparrow,
\]

while predictive loss remains close to the fixed fine-resolution reference.

### Phase C — reconvergence

The changed anchor returns to the original future distribution.

With decayed evidence, Cortex should again prefer a coarser, lower-complexity
field:

\[
\rho\uparrow,
\qquad
C(\rho)\downarrow.
\]

No split or merge command is supplied in any phase.

## 6. Candidate resolutions and frozen rule

The first campaign freezes

\[
\rho\in
\{0.03,0.06,0.12,0.25\},
\]

with

\[
\varepsilon=0.04
\]

nats of worst-region prequential log-loss tolerance.

These values are protocol constants for the first campaign. They must not be
changed after inspecting the benchmark answers without declaring a new
protocol.

## 7. What success means

A successful finite campaign requires:

1. coarse resolution dominates the equivalent-future phase;
2. resolution sharpens after one future diverges;
3. adaptive prediction materially outperforms always-coarse prediction in the
   divergent phase;
4. adaptive loss remains near the always-fine reference;
5. field complexity increases during divergence;
6. resolution and complexity fall again after reconvergence.

This would support:

\[
\boxed{
\text{Cortex can allocate representational complexity according to predictive
need rather than fixed ontology}.
}
\]

## 8. Non-claims and complexity warning

The controller searches a small fixed grid of resolutions and computes
effective rank repeatedly. It is therefore a research reference, not a native
hot-path design.

A scalable system should avoid repeated full kernel eigendecompositions and may
need local rather than global resolution.

The experiment does not establish optimal predictive rate-distortion, nor does
it prove that effective rank is the correct universal rate measure.


## 9. Campaign result

The frozen 20-seed campaign passed without changing the protocol constants.

### Resolution and complexity

| Phase | Mean selected \(\rho\) | Coarsest fraction | Finest-two fraction | Mean effective rank |
|---|---:|---:|---:|---:|
| Equivalent futures | **0.250** | **1.000** | — | **1.022** |
| Divergent future | **0.113** | — | **0.384** | **2.120** |
| Reconverged futures | **0.250** | **1.000** | — | **1.020** |

Thus the controller spends minimal predictive complexity when the three anchors
have equivalent futures, adds structure when one future diverges, and returns
to the coarse field after reconvergence.

Every one of the 20 seeds finished at

\[
\rho=0.25.
\]

### Predictive distortion during divergence

Mean prequential negative log loss:

| Readout | NLL |
|---|---:|
| Always coarse \(\rho=0.25\) | 0.807 |
| **Adaptive rate-distortion** | **0.769** |
| Always fine \(\rho=0.03\) | 0.757 |

The adaptive controller therefore reduces divergent-phase NLL by approximately

\[
0.038
\]

nats relative to always-coarse prediction while remaining only approximately

\[
0.012
\]

nats above the always-fine reference.

Across individual seeds, adaptive prediction beat always-coarse prediction in
19 of 20 runs and remained within 0.08 nats of always-fine prediction in all
20 runs.

The aggregate adaptive improvement recovers most of the predictive benefit of
fine resolution without maintaining fine complexity in the equivalent and
reconverged phases.

### Structural interpretation

The finite result supports the intended constrained objective:

\[
\boxed{
\text{use additional representational complexity only when coarse structure
causes measurable predictive distortion}.
}
\]

In the equivalent phase, all candidate resolutions make sufficiently similar
predictions, so effective-rank minimization selects the coarse field.

When one regional future diverges, its prequential loss makes the coarse
representation inadmissible often enough that the controller sharpens.

When the futures reconverge and decayed evidence forgets the obsolete
distinction, the coarse representation again enters the distortion band and is
selected because its effective rank is lower.

No external phase label or split/merge instruction enters the controller.

## 10. Stability and implementation caveat

The first reference is deliberately reactive. During the divergent phase it
changed selected resolution approximately 29.6 times on average over the full
phase, versus about one transition in the equivalent and reconverged phases.

This does not invalidate the predictive result, but it identifies a real
engineering/research target:

- hysteresis or minimum dwell time;
- local rather than global resolution;
- cached/incremental complexity estimates.

These should be introduced as a new protocol rather than tuned into the frozen
v1.7-R result after observing the benchmark.

The production hot path remains unchanged.
