# Cortex v1.11-R — Fused Support-Sparse Predictive Application

Status: **research optimization / not a production release**.

v1.10-R measured the existing v1.9-R sparse-resolution implementation across
9, 18, 36, 72, and 144 anchors. Sparse maintenance first exceeded 1.10x over
dense sampled maintenance at 72 anchors and reached about 1.28x at 144 anchors.
The preregistered expectation that sampled SVD would already provide a 1.30x
benefit at 36 anchors was not supported; the measured benefit was only about
1.05x.

That result redirects optimization toward the remaining dense predictive
application.

## 1. Remaining dense operation

For perceptual responsibilities

\[
w\in\Delta^{A-1},
\]

candidate prediction applies

\[
a_r = w^\top K_{\rho_r}
\]

for each of \(m\) candidate resolutions. Even if kernel geometry itself is
maintained sparsely, applying the full candidate bank remains approximately

\[
O(mA^2)
\]

per observation.

The local readout similarly evaluates

\[
a_{\rm loc}
=
w^\top K^{\rm loc}.
\]

v1.11-R asks whether perceptual locality can reduce this application cost.

## 2. Exact fusion first

Before introducing approximation, the first optimization computes perceptual
responsibilities only once per observation and reuses them for:

- candidate predictive readout;
- local predictive readout;
- regional prequential-loss attribution;
- field outcome evidence update.

The fused full-support mode must reproduce the validated v1.9-R sparse path
numerically and structurally.

This separates ordinary duplicate-work removal from support approximation.

## 3. Retained-mass support

Let \(S\) be the smallest set of anchors, ordered by responsibility, whose
total mass satisfies

\[
M_S
=
\sum_{i\in S} w_i
\ge q.
\]

The first protocol fixes

\[
\boxed{q=0.999}.
\]

The retained responsibilities are renormalized:

\[
\widetilde w_i
=
\begin{cases}
w_i/M_S, & i\in S,\\
0, & i\notin S.
\end{cases}
\]

The discarded mass is

\[
\delta=1-M_S.
\]

A useful exact property is

\[
\boxed{
\operatorname{TV}(w,\widetilde w)=\delta\le 1-q.
}
\]

Indeed, the \(L^1\) error inside the retained support equals \(\delta\), and
the omitted support contributes another \(\delta\).

This is a bound on perceptual responsibility itself. Because the predictive
kernel is subsequently normalized, it is not asserted as a universal bound on
final outcome-distribution error; that quantity is audited empirically.

## 4. Support-sparse readout

Candidate activation becomes

\[
\widetilde a_r
=
\widetilde w_S^\top
K_{\rho_r}[S,:],
\]

with cost approximately

\[
O(mkA),
\qquad
k=|S|.
\]

The local operator is evaluated from the same retained rows:

\[
\widetilde a_{\rm loc}
=
\sum_{i\in S}
\widetilde w_i
K_{\rho_{c_i}}[i,:].
\]

The first reference uses a deterministic stable sort to obtain the smallest
retained-mass support. The sort contributes \(O(A\log A)\), still below the
dense \(O(A^2)\) application when support remains meaningfully smaller than
the full field.

## 5. Exact world-model updates

Support compression affects **predictive readout only**.

After the outcome arrives, Cortex updates regional loss attribution and field
outcome counts using the original full responsibility vector \(w\), not
\(\widetilde w\).

Thus semantic evidence accumulation is not sparsified in v1.11-R. Any observed
behavioral difference is attributable to prequential predictive compression,
not to a different learned world state.

## 6. Campaign

Protocol: `support-sparse-prediction-v1`.

The campaign evaluates

\[
A\in\{36,72,144,288\}
\]

with three paired modes:

1. **legacy sparse** — validated v1.9-R sparse controller;
2. **fused full support** — one perceptual pass, \(q=1\);
3. **support sparse** — fused controller with \(q=0.999\).

Each case uses the same chronological observations and outcomes.

The campaign records:

- candidate and local prediction parity for exact fusion;
- total-variation error for support-compressed predictions;
- local-resolution-state disagreement;
- divergent-phase NLL;
- reconvergence behavior;
- mean/max retained support fraction;
- mean/max discarded responsibility mass;
- paired wall-time speedups.

## 7. Validity conditions

The campaign is considered behaviorally valid only if:

- fused full-support candidate and local predictions match v1.9-R within
  \(10^{-10}\);
- fused full-support local resolution states agree exactly;
- actual discarded responsibility mass never exceeds \(0.001\) beyond
  floating-point tolerance;
- at every anchor scale, median mean prediction TV is at most 0.01 and median
  maximum prediction TV at most 0.05;
- median local-resolution disagreement is at most 5%;
- divergent NLL is no more than 0.02 nats above fused full support;
- at least 95% of reconverged-tail states are fully coarse.

These are validity constraints. They are allowed to fail the CI campaign.

## 8. Performance hypotheses

Performance outcomes are reported but do not invalidate an otherwise correct
characterization.

The first protocol tests whether:

- mean support fraction is at most 50% at 144 anchors;
- mean support fraction is at most 40% at 288 anchors;
- support compression adds at least 1.15x over exact fusion at 144 anchors;
- support compression adds at least 1.25x over exact fusion at 288 anchors;
- combined fusion + support compression reaches at least 1.25x over the v1.9-R
  path at 144 anchors and 1.40x at 288 anchors.

These thresholds are hypotheses, not guarantees. A failed threshold remains a
negative result and must not be weakened post-hoc under the same protocol ID.

## 9. Complexity warning

This optimization does not yet make perceptual matching itself sparse.
Computing all anchor responsibilities remains \(O(AD)\) for feature dimension
\(D\), and the support is selected by sorting all \(A\) responsibilities.

If retained support remains small and predictive application becomes cheaper,
the next plausible target is approximate nearest-anchor retrieval or a
hierarchical support index. That step would affect perception itself and
therefore requires a separate error contract.
