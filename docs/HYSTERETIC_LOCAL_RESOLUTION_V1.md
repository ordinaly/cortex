# Cortex v1.8-R — Hysteretic Local Adaptive Resolution

Status: **research model / not a production release**.

Cortex v1.7-R established a predictive rate-distortion controller that selects
the lowest-complexity global semantic resolution whose recent worst-region
prequential loss lies within a fixed tolerance of the best candidate.

The frozen v1.7-R campaign passed, but its divergent phase exposed two limits:

1. the reactive controller can switch resolution repeatedly near the
   admissibility boundary;
2. one difficult region can force the entire predictive field to use a finer
   resolution.

v1.8-R addresses these limits without modifying the production runtime or the
frozen `spec-v0.9.5` executable specification.

## 1. Hysteresis

Let the candidate resolutions be ordered from fine to coarse,

\[
\rho_1 < \rho_2 < \cdots < \rho_m,
\]

and let

\[
D_t(\rho)
\]

be the v1.7-R worst-supported-region prequential distortion.

The original controller uses one tolerance \(\varepsilon\). v1.8-R introduces
a symmetric hysteresis margin \(h\),

\[
\varepsilon_{\rm in}
=
\max(0,\varepsilon-h),
\qquad
\varepsilon_{\rm out}
=
\varepsilon+h.
\]

For current resolution \(\rho_t\), define

\[
D_t^\star=\min_\rho D_t(\rho).
\]

The controller holds its current state while

\[
D_t(\rho_t)
\le
D_t^\star+\varepsilon_{\rm out}.
\]

If the current state leaves the outer band, Cortex sharpens immediately to the
coarsest finer candidate that restores the original v1.7-R constraint,

\[
D_t(\rho)
\le
D_t^\star+\varepsilon.
\]

Coarsening is deliberately slower. A coarser candidate must lie inside the
inner band,

\[
D_t(\rho)
\le
D_t^\star+\varepsilon_{\rm in},
\]

for \(T_{\downarrow}\) consecutive evidence states before promotion.

The reference protocol fixes

\[
\varepsilon=0.04,
\qquad
h=0.01,
\qquad
T_{\downarrow}=12.
\]

Repeated calls to the decision function on the same evidence snapshot do not
advance the dwell counter.

The intended behavior is asymmetric: predictive failure can create detail
quickly, while removing detail requires persistent evidence that the
distinction is no longer needed.

## 2. Local adaptive resolution

The global controller still assigns one \(\rho_t\) to every anchor.

v1.8-R instead permits a vector

\[
\boldsymbol{\rho}_t
=
(\rho_{t,1},\ldots,\rho_{t,A}),
\]

where \(A\) is the number of perceptual anchors.

Each anchor uses the same prequential candidate-loss evidence already tracked
by v1.7-R, but applies the hysteresis rule to its own regional loss column.

This creates a local resolution state without requiring an externally supplied
split, phase label, or ontology edit.

## 3. Row-adaptive predictive operator

A naive pairwise variable-bandwidth Gaussian need not preserve the positive
semidefinite structure used by the global effective-rank surrogate.

The reference therefore uses a row-adaptive operator. Let \(d_{ij}\) be the
predictive semantic distance between anchors \(i\) and \(j\). Define

\[
\boxed{
K^{\rm loc}_{ij}
=
\exp\left[
-\frac{1}{2}
\left(
\frac{d_{ij}}{\rho_i}
\right)^2
\right].
}
\]

Interpretation: anchor \(i\) controls how broadly its activation spreads
through predictive semantic space.

For perceptual responsibilities \(w(o)\), local field activation is

\[
a(o)=w(o)^\top K^{\rm loc},
\]

followed by the same normalization and outcome readout as the fuzzy predictive
field.

When every anchor has the same resolution, \(\rho_i=\rho\) for all \(i\),
the operator reduces exactly to the v1.7-R global predictive kernel.

## 4. Local complexity

Because \(K^{\rm loc}\) is generally non-symmetric, eigenvalue effective rank
is not the appropriate direct surrogate.

Let \(\sigma_1,\ldots,\sigma_r\) be its singular values and

\[
p_k=
\frac{\sigma_k}{\sum_j\sigma_j}.
\]

v1.8-R uses the effective singular rank

\[
\boxed{
C_{\rm loc}
=
\exp\left(
-\sum_k p_k\log p_k
\right).
}
\]

For a uniform resolution, the predictive kernel is symmetric positive
semidefinite, so this agrees with the existing effective-rank surrogate up to
numerical precision.

## 5. Important approximation

The local controller does **not** exhaustively score every possible local
resolution vector. That search has size

\[
m^A
\]

for \(m\) candidate resolutions and \(A\) anchors, which would immediately
reintroduce combinatorial cost.

Instead, each region uses its prequential loss under the same finite set of
uniform candidate resolutions as a diagnostic for whether that region needs
more or less semantic detail. Those regional decisions then parameterize the
row-adaptive operator.

This is a bounded research approximation, not a claim of globally optimal
local rate-distortion.

## 6. Sparse-divergence campaign

The v1.8-R campaign expands the field from three to nine perceptually related
anchors.

Three phases are used:

1. **equivalent** — all anchors share the same future distribution;
2. **divergent** — exactly one anchor develops a different future;
3. **reconverged** — that anchor returns to the shared future.

The benchmark compares v1.7-R reactive global resolution, v1.8-R hysteretic
global resolution, v1.8-R hysteretic local resolution, fixed finest resolution,
and fixed coarsest resolution.

No controller receives the phase label or the identity of the divergent
anchor.

## 7. Frozen protocol constants

The first committed campaign fixes

\[
\rho
\in
\{0.03,0.06,0.12,0.25\},
\]

\[
\varepsilon=0.04,
\qquad
h=0.01,
\qquad
T_{\downarrow}=12,
\]

with loss decay \(0.995\), field evidence decay \(0.998\), perceptual
temperature \(0.03\), observation noise \(0.20\), and nine anchors.

Per seed: 250 equivalent cycles, 400 divergent cycles, and 500 reconverged
cycles. The frozen campaign uses 20 seeds.

Exploratory development smoke tests existed before these constants were
committed, so this is not presented as a blinded preregistration. The committed
constants and gates must nevertheless remain fixed for the named
`hysteretic-local-resolution-v1` protocol. Any later tuning requires a new
protocol identifier.

## 8. Success gates

The 20-seed campaign is considered successful only if all of the following
hold on aggregate:

1. divergent-phase hysteretic global switching is at most 50% of the reactive
   v1.7-R switching count;
2. hysteretic global divergent NLL is no more than 0.02 nats above reactive
   global NLL;
3. hysteretic local divergent NLL is no more than 0.04 nats above hysteretic
   global NLL;
4. local divergent effective singular rank is at most 75% of hysteretic global
   effective rank;
5. no more than 35% of anchor-time is refined during the tail of the divergent
   phase;
6. at least 95% of anchor-time is at the coarsest resolution during the tail of
   the reconverged phase.

These gates separately test

\[
\text{hysteresis}
\Rightarrow
\text{less switching without material predictive loss},
\]

and

\[
\text{local resolution}
\Rightarrow
\text{predictive detail is concentrated where it is needed}.
\]

## 9. Complexity warning

The reference remains intentionally expensive.

For every observation it still evaluates all uniform candidate resolutions,
and local complexity currently uses a full singular-value decomposition of the
row-adaptive operator.

This is suitable for a finite research campaign, not a production hot path.

If v1.8-R passes, the next implementation target is **cached / incremental local
complexity estimation** plus a sparse update rule that only revisits regions
whose predictive evidence changed materially.

## 10. Non-claims

v1.8-R does not establish globally optimal predictive rate-distortion, optimal
hysteresis constants, a universal information-theoretic complexity measure,
production scalability, a general solution to adaptive mesh/topological
refinement, or superiority to neural architectures outside the declared
benchmark.

The research claim under test is narrower:

\[
\boxed{
\text{Cortex can make representational precision stateful and local, allocating
detail only where persistent predictive evidence requires it.}
}
\]
