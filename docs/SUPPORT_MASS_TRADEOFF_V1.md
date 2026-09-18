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


## 7. Campaign result

The three-seed sweep completed successfully at 72, 144, and 288 anchors. Every
tested retained mass was behaviorally admissible on every seed.

### Support fraction

Median mean retained support fractions were:

| anchors | q=0.99 | q=0.995 | q=0.999 |
|---:|---:|---:|---:|
| 72 | 0.649 | 0.730 | 0.868 |
| 144 | 0.649 | 0.729 | 0.864 |
| 288 | **0.641** | 0.722 | 0.859 |

Thus lowering retained mass from 0.999 to 0.99 does materially reduce the
number of applied source rows, to roughly 64% of anchors at 288 anchors.

### Fidelity

Despite the more aggressive compression, prediction drift remained very small.

At 288 anchors with \(q=0.99\):

- mean local prediction TV: **4.03e-7**;
- maximum local prediction TV: **1.78e-6**;
- mean candidate prediction TV: **6.46e-6**;
- maximum candidate prediction TV: **7.28e-5**;
- local-resolution disagreement: **0.0**;
- divergent NLL: **0.8460234** versus **0.8460232** full support;
- reconverged-tail fully coarse fraction: **1.0**;
- maximum discarded responsibility mass: **0.009999**.

Every declared admissibility condition passed for every tested mass and seed.

### Performance

The speed result is much weaker:

| anchors | q=0.99 | q=0.995 | q=0.999 |
|---:|---:|---:|---:|
| 72 | 0.974x | 0.973x | 0.970x |
| 144 | 0.994x | 0.992x | 0.985x |
| 288 | **1.050x** | 1.038x | 1.028x |

The preregistered support-fraction hypothesis for \(q=0.99\) at 288 anchors
passed, but the 1.10x speed hypothesis did not. The 1.05x hypothesis for
\(q=0.995\) also did not pass.

The fastest mass that was admissible on every 288-anchor seed was

\[
\boxed{q=0.99},
\]

but its gain over full support was only about 5%.

## 8. Interpretation

v1.12-R shows that retained-mass truncation is **not constrained by predictive
fidelity** in this fixture. It is constrained by implementation economics.

Reducing source support from 100% to about 64% does not translate into a
comparable runtime reduction. Stable sorting, support extraction/fancy
indexing, row materialization, and other non-support-dependent work absorb most
of the theoretical saving.

Therefore Cortex should not promote support truncation as the next default
optimization merely because it is mathematically controlled.

The current evidence supports:

\[
\boxed{
\text{keep exact fused full-support prediction as the research baseline}
}
\]

and treat retained-mass compression as an optional technique that may become
useful only after support discovery itself is made cheaper.

## 9. Next step

The next research increment should perform stage attribution inside the fused
prequential step before another optimization is chosen.

At minimum it should separately measure:

1. perceptual responsibility computation;
2. hysteresis/decision work;
3. candidate-kernel application;
4. local predictive application;
5. regional loss update;
6. field outcome update;
7. semantic/kernel cache refresh;
8. sampled complexity/SVD work.

This will determine whether the next target should be perceptual indexing,
kernel application, cache maintenance, or state-update bookkeeping rather than
inferring the bottleneck from asymptotic form alone.
