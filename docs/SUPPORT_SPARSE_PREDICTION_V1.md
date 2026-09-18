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


## 10. Campaign result

The three-seed campaign completed successfully at 36, 72, 144, and 288
anchors. All behavioral validity conditions passed.

### Exact fusion

The full-support fused controller reproduced the v1.9-R sparse path to floating
point precision:

- maximum candidate-prediction error: approximately \(2.2\times10^{-16}\);
- maximum local-prediction error: approximately \(2.2\times10^{-16}\);
- local-resolution-state disagreement: **0.0** at every tested scale.

Despite preserving behavior, fusion produced a large speedup:

| anchors | v1.9-R legacy | fused full support | speedup |
|---:|---:|---:|---:|
| 36 | 1465.3 us/step | 672.7 us/step | **2.184x** |
| 72 | 2540.8 us/step | 1055.5 us/step | **2.459x** |
| 144 | 4712.0 us/step | 1843.2 us/step | **2.550x** |
| 288 | 9505.1 us/step | 3955.6 us/step | **2.403x** |

Thus the dominant optimization in v1.11-R is not perceptual support truncation;
it is elimination of duplicate perceptual and predictive work inside one
prequential step.

### 99.9% retained-mass support

The support-compressed path preserved predictive behavior extremely well.

At 144 anchors:

- mean local prediction TV: \(1.57\times10^{-7}\);
- maximum local prediction TV: \(9.48\times10^{-7}\);
- local-resolution disagreement: **0.0**;
- divergent NLL: **0.7046765** versus **0.7046764** for full support;
- reconverged-tail fully coarse fraction: **1.0**.

At 288 anchors:

- mean local prediction TV: \(5.49\times10^{-8}\);
- maximum local prediction TV: \(3.16\times10^{-7}\);
- local-resolution disagreement: **0.0**;
- divergent NLL: **0.8446255** for both paths to the shown precision;
- reconverged-tail fully coarse fraction: **1.0**.

The exact responsibility-mass contract was also respected: maximum discarded
mass remained below 0.001 in every case.

### Support sparsity hypothesis: not supported

The 0.999 retained-mass target was too conservative for this perceptual
geometry. Mean retained support fractions were approximately:

| anchors | mean support fraction |
|---:|---:|
| 36 | **0.876** |
| 72 | **0.875** |
| 144 | **0.872** |
| 288 | **0.868** |

The support therefore remains close to dense even as anchor count grows.

As a consequence, support compression itself did not improve runtime:

| anchors | support vs fused |
|---:|---:|
| 36 | 0.964x |
| 72 | 0.964x |
| 144 | 0.987x |
| 288 | 1.032x |

The preregistered support-fraction and support-speedup hypotheses at 144 and 288
anchors therefore failed. Their thresholds are retained unchanged as negative
results.

### Combined result

Because exact fusion is already large, the support path still remains much
faster than the old v1.9-R execution path:

- **2.522x** at 144 anchors;
- **2.483x** at 288 anchors.

These gains should be attributed primarily to exact fusion, not to support
compression.

## 11. Next implication

v1.11-R changes the optimization priority again.

The validated result supports promoting **one-pass prequential fusion** as the
new research baseline.

The retained-mass idea remains mathematically controlled but is not useful at
\(q=0.999\) because the perceptual distribution is too diffuse. The next
experiment should not silently lower \(q\) under this protocol. Instead, a new
protocol should sweep more aggressive retained masses, for example

\[
q\in\{0.99,0.995,0.999\},
\]

and measure the full prediction-error/speed/support tradeoff.

If useful speedup only appears after discarding enough mass to damage predictive
behavior, support truncation should be abandoned in favor of a different
perceptual acceleration strategy, such as indexed approximate neighbor
retrieval or hierarchical anchor organization.
