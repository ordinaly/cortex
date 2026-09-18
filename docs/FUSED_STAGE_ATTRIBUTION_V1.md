# Cortex v1.13-R — Fused Prequential Stage Attribution

Status: **research performance characterization / not a production release**.

v1.11-R established exact one-pass prequential fusion as the strongest recent
optimization, producing roughly 2.2x-2.55x speedup over the earlier sparse
execution path while preserving behavior to floating-point precision.

v1.12-R then showed that responsibility truncation is behaviorally safe but not
economically compelling in the current Python/NumPy implementation: even at
288 anchors, reducing mean support to about 64% yielded only about 1.05x over
fused full support.

v1.13-R therefore measures the fused path internally before another
optimization is chosen.

## 1. Profiled stages

The campaign attributes the full-support fused prequential transition to:

1. **hysteresis scan** — local resolution transition checks;
2. **complexity** — sampled effective singular-rank telemetry;
3. **perceptual** — full perceptual responsibility computation;
4. **candidate readout** — application of all candidate predictive kernels;
5. **local readout** — application of the current row-adaptive local operator;
6. **tracker update** — regional prequential-loss update;
7. **field update** — outcome evidence update;
8. **cache refresh** — semantic feature/probability refresh and dirty geometry
   maintenance.

The profiler also records unattributed timing remainder and requires high
accounting coverage.

## 2. Reference semantics

The profiled candidate and local readout helpers are differential-tested against
the v1.11-R fused full-support bundle.

The campaign does not introduce a new approximation or adaptive-resolution
rule.

## 3. Scale sweep

Protocol: `fused-stage-attribution-v1`.

The campaign evaluates

\[
A\in\{72,144,288\}
\]

over paired deterministic synthetic streams and three seeds.

For each stage it reports median:

- microseconds per step;
- share of total instrumented step time.

It also reports kernel-row refresh duty, regional scan duty, and SVD duty.

## 4. Interpretation rule

No stage is preregistered as the expected winner.

The next optimization target should be the stage that is both:

- a large measured fraction of fused runtime at the largest tested scale; and
- semantically separable enough to optimize without weakening the validated
  predictive/adaptive contracts.

This avoids optimizing according to asymptotic intuition when constant factors
or NumPy vectorization dominate the actual research implementation.

## 5. Timing caveat

Fine-grained `perf_counter_ns` instrumentation adds overhead. Absolute
microsecond values should therefore not be compared directly with uninstrumented
campaign timings.

Stage shares and scaling trends are the primary quantities. The campaign records
the accounted fraction so instrumentation gaps remain explicit.
