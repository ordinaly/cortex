# Cortex v1.19-R — Low-Overhead Fused-Stage Crossover Profile

Status: **frozen profiling protocol / development passed / official campaign untouched**.

Protocol ID: **low-overhead-fused-stage-profile-v1.19-R**.

## Motivation

v1.18-R showed that fine-grained within-cache attribution can materially
overstate the end-to-end payoff of a micro-optimization. The kernel ladder
looked large inside the instrumented cache but failed all frozen official
large-scale performance gates.

v1.19-R therefore measures the current optimized research path with much lower
observer overhead before selecting another target.

## Measurement design

Each stage is measured in its **own full replay**.

For a given seed/scale:

1. create a fresh controller;
2. execute the full deterministic trace;
3. time only one target stage with one timer pair per step;
4. execute every other stage untimed;
5. repeat from fresh state for each stage;
6. run one additional replay timing only the complete step.

Stage order rotates by seed to reduce systematic runner-load bias.

Compared with the previous profiler, no run contains eight internal timer
pairs per step.

## Stages

- hysteresis/state scan;
- complexity refresh;
- perceptual responsibilities;
- candidate readout;
- local readout;
- tracker update;
- field update;
- cache refresh.

## Semantic parity

Every stage-specific replay must produce the same prediction checksum as the
total-step replay for the same seed and anchor count.

Timing instrumentation must therefore remain observational.

## Scale sweep

Anchor counts: **72, 144, 288**.

Development seeds: **0..5**.

Reserved untouched official seeds: **600..604**.

## Metrics

For each scale report median:

- total fused-step microseconds;
- microseconds per step for every stage;
- stage / total ratio from independent replays;
- dominant measured stage;
- kernel-row refresh fraction;
- anchor-scan fraction;
- SVD refresh fraction.

The sum of stage ratios is a diagnostic only because stages are measured in
independent replays; it is not an exact same-run accounting identity.

## Development validity gates

Before official evaluation:

1. all prediction-checksum discrepancies <= **1e-12**;
2. all stage and total timings are finite and non-negative;
3. total timing is positive;
4. diagnostic refresh/scan fractions lie in **[0,1]**;
5. exactly the declared development seeds and anchor scales are present.

There is deliberately no preregistered dominant stage.

## Interpretation

The official campaign chooses the next optimization target from the coarse
stage ordering.

If cache refresh remains clearly dominant, the next cache intervention should
target writes/layout rather than kernel math. If candidate readout dominates,
the engineering line should move there. If ordering is unstable or close,
freeze the optimization line rather than chase timer noise.

## Claim boundary

This is a Python/NumPy research attribution campaign on declared synthetic
traces. It does not characterize the native Rust runtime.

## Development outcome before freeze

The 18-case development campaign passed every validity gate with **0**
prediction-checksum discrepancy.

At 288 anchors, median low-overhead stage shares were:

- cache refresh: **37.23%**;
- candidate readout: **21.14%**;
- hysteresis/state scan: **18.30%**;
- complexity refresh: **13.51%**;
- local readout: **3.63%**;
- tracker update: **2.90%**;
- perceptual computation: **2.69%**;
- field update: **0.29%**.

The independent-replay stage/total ratio summed to about **99.80%**, supporting
the consistency of the coarse measurement method without treating the sum as a
same-run accounting identity.

No stage winner is frozen in advance; the untouched official campaign chooses
the next target.
