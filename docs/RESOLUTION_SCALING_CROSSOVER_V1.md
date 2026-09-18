# Cortex v1.10-R — Resolution Scaling Crossover

Status: **research performance characterization / not a production release**.

Cortex v1.9-R established two useful facts on the nine-anchor fixture:

1. an exact shared semantic/kernel cache reproduces the validated v1.8-R path
   exactly on the declared differential campaign while running about 1.58x
   faster;
2. sparse geometry refresh, sparse regional scans, and sampled complexity
   telemetry preserve the same behavior to high precision and increase the
   measured speedup to about 1.67x.

At nine anchors, however, the sparse layer itself contributes only a modest
additional speedup over exact caching.

v1.10-R therefore does **not** add another optimization mechanism. It measures
where the existing mechanisms become worthwhile as field size grows.

## 1. Cost decomposition

The campaign separates three execution modes.

### Exact SVD

`exact-svd` uses the exact shared v1.9-R cache but computes effective singular
rank on every observation.

This retains the research semantics of the v1.8-R complexity trace.

### Dense sampled

`dense-sampled` rebuilds semantic geometry and the full candidate-kernel bank
exactly on every observation and scans all active anchors, but computes
complexity only every 32 observations unless the local resolution signature
changes.

Comparing this mode with `exact-svd` isolates the cost of repeated SVD.

### Sparse

`sparse` uses the accepted v1.9-R sparse settings:

\[
\delta_F=0.0025,
\qquad
\delta_L=0.001,
\qquad
T_{\rm scan}=64,
\qquad
T_{\rm svd}=32.
\]

Comparing it with `dense-sampled` isolates the benefit of incremental geometry
maintenance and sparse hysteresis scans after the SVD confound is removed.

## 2. Anchor sweep

The first campaign evaluates

\[
A\in\{9,18,36,72,144\}.
\]

Every case uses the same feature dimension, candidate resolutions, perceptual
temperature, outcome alphabet, and total number of chronological observations.

The exact-SVD mode is run only through 36 anchors. Beyond that point the purpose
of the campaign is to compare dense versus sparse maintenance after complexity
sampling; paying an exact cubic SVD at every step would add cost without
answering the kernel-maintenance crossover question.

## 3. Behavioral audit

For anchor counts where exact-SVD is run, `dense-sampled` must retain exact
prediction and resolution-state parity because complexity telemetry is not a
control input.

At every scale the sparse path is compared with `dense-sampled` using:

- mean and maximum total-variation prediction error;
- local-resolution-state disagreement fraction;
- kernel-row refresh duty;
- regional scan duty;
- SVD duty.

## 4. Performance question

The main quantity is

\[
S_A
=
\frac{
T_{\rm dense-sampled}(A)
}{
T_{\rm sparse}(A)
}.
\]

The campaign reports the first tested anchor count for which

\[
S_A\ge1.10.
\]

This is an empirical crossover for the declared Python/NumPy research
implementation, **not** an asymptotic theorem and not automatically transferable
to the Rust runtime.

## 5. Gates

The campaign must preserve behavioral fidelity at every tested scale.

Additionally:

- sampled complexity must provide at least 1.30x speedup over exact SVD at
  36 anchors;
- sparse maintenance must provide at least 1.15x speedup over dense sampled
  maintenance at 144 anchors;
- at 144 anchors, kernel-row refresh duty must be at most 35%, regional scan
  duty at most 60%, and SVD duty at most 10%.

No small-anchor sparse speedup is required. If dense vectorization wins at small
sizes, that is a useful result rather than a failure of the architecture.

## 6. Why this precedes deeper optimization

The remaining dense work includes full semantic-feature evaluation, exact
outcome-probability refresh, and all-anchor perceptual weighting.

Those can be made lazier or sparser, but doing so adds state machinery and
approximation risk.

The scaling sweep tells us whether the existing \(O(A^2)\) geometry path or the
remaining \(O(A)\) field-statistics path is actually limiting performance in
the range relevant to Cortex. The next implementation step should follow that
measurement rather than complexity intuition alone.
