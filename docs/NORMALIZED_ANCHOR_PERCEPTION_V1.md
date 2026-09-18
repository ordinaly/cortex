# Cortex v1.14-R — Normalized-Anchor Perceptual Fast Path

Status: **research optimization / not a production release**.

v1.13-R attributed roughly 49%-54% of fused prequential runtime to perceptual
responsibility computation across 72-288 anchors.

Inspection of the research field exposed an exact redundancy.

## 1. Existing invariant

`FuzzyPredictiveField.__init__` stores anchors as

\[
a_i
\leftarrow
\frac{a_i}{\lVert a_i\rVert}.
\]

Therefore every stored anchor is already normalized under the field's own
constructor invariant.

The previous `possibility()` implementation nevertheless delegated to a
generic helper that normalized every anchor row again on every observation.

That generic behavior is appropriate for arbitrary external anchor arrays, but
it is redundant inside a constructed field.

## 2. Fast path

The optimized field normalizes only the incoming observation,

\[
\widehat x
=
\frac{x}{\lVert x\rVert},
\]

then computes

\[
d_i
=
\max(0,1-a_i^\top\widehat x)
\]

directly against the stored normalized anchor matrix.

Possibility remains

\[
\pi_i
=
\exp(-d_i/T).
\]

The same finite-positive temperature validation is preserved.

The standalone generic `perceptual_possibility(...)` helper remains unchanged
for callers supplying arbitrary unnormalized anchor arrays.

## 3. Why this should be exact

For mathematically normalized anchors,

\[
\frac{a_i}{\lVert a_i\rVert}=a_i.
\]

The old and new paths therefore implement the same cosine-distance expression.
Only redundant floating-point normalization work is removed.

A paired legacy subclass is retained in the benchmark campaign to call the old
generic helper explicitly.

## 4. Campaign

Protocol: `normalized-anchor-perception-v1`.

The campaign compares legacy and fast fused full-support controllers at

\[
A\in\{72,144,288,576\}
\]

over identical chronological streams and three seeds.

It records:

- standalone perceptual-weight speedup;
- full fused-controller speedup;
- maximum candidate prediction error;
- maximum local prediction error;
- local-resolution-state disagreement;
- final outcome-count error;
- regional loss-state error;
- regional anchor-mass error.

## 5. Validity

The optimization is accepted as semantically exact only if all paired
prediction/state errors remain at or below \(10^{-10}\) and structural
resolution decisions agree exactly.

Performance thresholds are reported as hypotheses rather than correctness
conditions.

## 6. Performance hypotheses

The first campaign tests whether:

- perceptual responsibility computation is at least 5x faster at 288 anchors;
- whole fused execution is at least 1.30x faster at 288 anchors;
- whole fused execution is at least 1.30x faster at 576 anchors.

A failed performance hypothesis remains a measured negative result and does not
weaken the semantic parity requirement.

## 7. Scope

This change optimizes the Python research field only. It does not alter the
native Rust runtime or frozen `spec-v0.9.5`.

If the fast path produces the expected exact speedup, it becomes the research
baseline and the next attribution campaign should be rerun because removing a
~50% stage can expose a different dominant bottleneck.
