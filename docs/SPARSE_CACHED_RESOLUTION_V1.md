# Cortex v1.9-R — Sparse Cached Local Resolution

Status: **research optimization / not a production release**.

Cortex v1.8-R established hysteretic local adaptive resolution under the frozen
`hysteretic-local-resolution-v1` campaign. Its 20-seed result reduced mean
divergent-phase switching from 32.35 to 2.65, kept predictive NLL essentially
unchanged, reduced local effective complexity from 4.793 to 1.623, refined only
12.3% of anchor-time during divergence, and returned to a fully coarse field
after reconvergence.

v1.9-R asks a different question:

> Can the same adaptive behavior be computed with substantially less redundant
> semantic geometry, kernel, regional scan, and complexity work?

The validated v1.8-R implementation remains the reference oracle.

## 1. Exact shared-state cache

In v1.8-R every candidate resolution independently calls the field prediction
path. That repeatedly reconstructs semantic features, predictive distances,
kernels, and outcome probabilities from the same field state.

For one field state, v1.9-R computes

\[
F_t,
\qquad
D_t,
\qquad
P_t
\]

once, then constructs the finite candidate bank

\[
\mathcal K_t
=
\{K_{\rho_1},\ldots,K_{\rho_m}\}.
\]

For observation responsibility vector \(w\), every candidate activation is
then evaluated in one batched operation,

\[
A_r = w^\top K_{\rho_r}.
\]

The local operator needs no additional Gaussian construction. If anchor \(i\)
currently selects candidate index \(c_i\),

\[
K^{\mathrm{loc}}_{i,:}
=
K_{\rho_{c_i}}[i,:].
\]

Thus the local operator is assembled by row selection from the already computed
candidate bank.

This path is intended to be numerically equivalent to v1.8-R and is tested as a
differential oracle before any approximation is accepted.

## 2. Sparse semantic refresh

The approximate mode retains cached semantic features

\[
\widetilde F_t.
\]

After each field update, current features \(F_t\) are computed and anchor \(i\)
is declared geometry-dirty when

\[
\lVert F_t(i)-\widetilde F_t(i)\rVert_2
\ge
\delta_F.
\]

The first protocol fixes

\[
\delta_F=0.0025.
\]

Because drift is measured against the last **cached** feature rather than only
against the immediately previous feature, small changes accumulate and
eventually force refresh. A clean row therefore cannot drift indefinitely
without being revisited.

When a set \(S\) becomes dirty, only rows and columns incident to \(S\) are
recomputed for the semantic distance matrix and finite kernel bank. Outcome
probabilities remain exact and are refreshed every observation.

## 3. Sparse hysteresis scans

The regional prequential loss tracker remains exact.

For anchor \(a\), let

\[
L_t(:,a)
\]

be its vector of candidate regional losses. The anchor becomes loss-dirty when

\[
\lVert
L_t(:,a)-L_{t-1}(:,a)
\rVert_\infty
\ge
\delta_L
\]

with frozen first-protocol value

\[
\delta_L=0.001.
\]

The hysteresis transition rule is evaluated only for

- loss-dirty active anchors;
- anchors already inside a pending coarsening dwell;
- every active anchor on a periodic full safety scan.

The full-scan interval is

\[
T_{\mathrm{scan}}=64.
\]

This optimization is structurally well motivated: if an anchor receives no new
local contribution, exponential decay multiplies its regional loss numerator
and mass by the same factor, so their ratio is unchanged.

## 4. Cached complexity telemetry

v1.8-R computes a full singular-value decomposition whenever local complexity is
reported. Complexity does **not** feed the hysteresis transition rule.

v1.9-R therefore treats effective singular rank as sampled telemetry. It is
recomputed immediately when the local resolution signature changes and
otherwise at most every

\[
T_{\mathrm{svd}}=32
\]

observations.

The optimization campaign periodically computes an out-of-band exact SVD and
records the approximation error. This audit is excluded from the timed
controller path.

## 5. Three-way comparison

The optimization protocol `sparse-cached-resolution-v1` compares:

1. **reference** — validated v1.8-R implementation;
2. **cached** — exact shared-state cache, full scans, exact SVD;
3. **sparse** — shared cache + drift-triggered geometry refresh + sparse
   hysteresis scans + sampled complexity telemetry.

This separation prevents an approximate algorithmic change from being credited
for speedups that actually come from ordinary implementation reuse.

## 6. Acceptance gates

The exact cached path must satisfy strict differential parity:

- maximum prediction error \(\le 10^{-10}\);
- zero local-resolution-state disagreements;
- maximum complexity error \(\le 10^{-9}\).

It must also be at least 1.20x faster than the reference path on the paired CI
fixture.

The sparse path must satisfy:

- mean total-variation prediction error \(\le 0.01\);
- maximum total variation \(\le 0.05\);
- local resolution disagreement fraction \(\le 0.10\);
- divergent NLL no more than 0.03 nats above the exact cached path;
- at least 95% fully coarse states in the reconverged tail;
- kernel-row refresh fraction \(\le 0.50\);
- SVD duty fraction \(\le 0.10\);
- anchor scan fraction \(\le 0.60\);
- at least 1.50x speedup over the v1.8-R reference path.

These are engineering/research gates for this declared fixture, not universal
performance claims.

## 7. Complexity warning

The sparse implementation still computes current semantic features and outcome
probabilities for all anchors. Its savings come from avoiding repeated pairwise
distance/kernel construction, local Gaussian reconstruction, unnecessary
hysteresis scans, and per-observation SVD.

The next asymptotic step, if this campaign succeeds, is to move field statistics
themselves to an explicitly sparse/incremental representation and then profile
the crossover point where sparse bookkeeping beats dense vectorized NumPy.

That crossover must be measured rather than assumed: for small anchor counts,
dense vectorization can remain faster despite worse asymptotic work.

## 8. Non-claims

v1.9-R does not claim an optimal cache policy, an optimal drift threshold,
asymptotically optimal dynamic SVD, or production scalability.

The narrow claim under test is:

\[
\boxed{
\text{Cortex can preserve the validated v1.8-R local adaptive behavior while
avoiding most redundant resolution work.}
}
\]
