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


## 7. Campaign result

The three-seed characterization completed successfully at all five anchor
counts. Behavioral validity checks passed at every scale. One originally
declared performance hypothesis did not.

### Sparse crossover

Median paired timings were:

| anchors | dense sampled | sparse | sparse speedup |
|---:|---:|---:|---:|
| 9 | 654.2 us/step | 687.8 us/step | 0.950x |
| 18 | 954.0 us/step | 954.9 us/step | 0.999x |
| 36 | 1549.8 us/step | 1475.7 us/step | 1.049x |
| 72 | 2879.6 us/step | 2525.2 us/step | **1.140x** |
| 144 | 6134.9 us/step | 4784.3 us/step | **1.281x** |

The first tested case exceeding the declared 1.10x sparse-maintenance crossover
was therefore

\[
\boxed{A=72}.
\]

This establishes an empirical implementation crossover for the declared
Python/NumPy fixture. It is not an asymptotic theorem.

### Behavioral fidelity

Sparse approximation remained highly faithful across the sweep. At 144 anchors:

- mean total-variation prediction error: **1.99e-6**;
- maximum total variation: **8.12e-6**;
- local-resolution disagreement fraction: **0.0**;
- kernel-row refresh duty: **15.52%**;
- regional scan duty: **31.38%**;
- SVD duty: **3.32%**.

Thus the larger-scale speedup is not explained by a material behavioral change.

### SVD hypothesis: not supported

The protocol predicted at least a 1.30x benefit from sampled complexity SVD by
36 anchors. The observed median speedup was only

\[
\boxed{1.054\times}.
\]

The original check therefore **failed** and is retained as a negative result.
Its threshold is not weakened after observing the campaign.

At 9 and 18 anchors the sampled-SVD speedups were similarly modest, about
1.065x and 1.060x respectively.

This means the next optimization should not focus primarily on further SVD
engineering at these scales.

## 8. Consequence for the next optimization

The scaling data point to the remaining dense predictive application.

Even when semantic geometry is sparsely maintained, candidate prediction still
computes

\[
w^\top K_{\rho_r}
\]

against full \(A\times A\) kernels for every candidate resolution. The local
readout similarly multiplies the perceptual responsibility vector against the
full local operator.

If perceptual responsibility is concentrated on a small support, these products
can be reduced from approximately

\[
O(mA^2)
\]

per candidate bank application to

\[
O(mkA),
\qquad
k\ll A,
\]

by retaining a controlled perceptual support and applying only the corresponding
kernel rows.

The next research increment should therefore test **support-sparse predictive
application**, with explicit retained-mass/error bounds and differential
prediction audits. This is a better-supported target than additional SVD work.
