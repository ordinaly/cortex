# Cortex v1.12-R — Retained-Mass Support Tradeoff

Status: **research characterization / not a production release**.

v1.11-R established a strong exact optimization: fusing perceptual
responsibilities across candidate prediction, local prediction, regional-loss
attribution, and field update produced approximately 2.2x-2.55x speedup across
36-288 anchors with numerical parity to the previous v1.9-R execution path.

The same campaign also showed that retained mass

\[
q=0.999
\]

was too conservative as a sparsification mechanism. It retained roughly 87% of
anchors on average and produced essentially no additional speedup over fused
full-support prediction.

v1.12-R therefore does not revise that threshold post-hoc. It declares a new
tradeoff protocol and sweeps more aggressive retained masses.

## 1. Sweep

Protocol: `support-mass-tradeoff-v1`.

The campaign evaluates

\[
q\in\{0.99,0.995,0.999\}
\]

at

\[
A\in\{72,144,288\}.
\]

Each compressed run is paired with a fused full-support run over the same
chronological observations and outcomes.

## 2. Responsibility-space guarantee

For every retained mass \(q\), the support mechanism selects the smallest
descending-weight support \(S\) satisfying

\[
M_S\ge q
\]

and renormalizes retained responsibilities.

As established in v1.11-R,

\[
\operatorname{TV}(w,\widetilde w)
=
1-M_S
\le
1-q.
\]

The campaign verifies this mass contract directly for every run.

## 3. Behavioral admissibility

A seed/run is labeled **admissible** only if all of the following hold relative
to fused full support:

- mean local-prediction TV \(\le0.01\);
- maximum local-prediction TV \(\le0.05\);
- mean candidate-prediction TV \(\le0.01\);
- maximum candidate-prediction TV \(\le0.05\);
- local-resolution-state disagreement \(\le5\%\);
- divergent NLL increase \(\le0.02\) nats;
- reconverged-tail fully coarse fraction \(\ge95\%\);
- discarded responsibility mass obeys the declared \(1-q\) bound.

Admissibility is a measured outcome, not a CI requirement. A more aggressive
mass may legitimately fail and remain useful as a boundary point on the
tradeoff curve.

## 4. Performance measurements

For every \((A,q)\) pair the campaign records:

- full-support and compressed microseconds per step;
- paired speedup;
- mean and maximum retained support fraction;
- mean and maximum discarded mass;
- candidate and local predictive TV;
- resolution-state disagreement;
- divergent NLL;
- reconvergence behavior.

At 288 anchors, the campaign additionally identifies the **fastest retained
mass that is admissible on every seed**.

This selection rule is declared before observing the sweep.

## 5. Performance hypotheses

The first protocol tests, without making them validity conditions, whether:

- \(q=0.99\) reduces mean support below 70% at 288 anchors;
- \(q=0.99\) provides at least 1.10x speedup over fused full support at 288
  anchors;
- \(q=0.995\) provides at least 1.05x speedup at 288 anchors.

Failed hypotheses are retained as negative results.

## 6. Interpretation

There are three possible outcomes.

1. **A lower q is fast and admissible.**
   Freeze that operating point in a later protocol and test it more broadly.

2. **Lower q is admissible but still not faster.**
   The bottleneck is not the number of applied kernel rows; sorting/indexing or
   another dense component dominates.

3. **Useful speedup requires unacceptable predictive drift.**
   Abandon responsibility truncation and move to a structurally different
   acceleration, such as indexed/hierarchical anchor retrieval.

The point of v1.12-R is therefore to decide whether retained-mass sparsification
deserves further engineering at all.
