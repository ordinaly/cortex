# Cortex v1.14-R — Exact Perceptual Fast Path

Status: **frozen optimization protocol / official campaign passed**.

Protocol ID: `exact-perceptual-fast-path-v1.14-R`.

## Motivation

The v1.13-R fused-stage attribution campaign identified perceptual
responsibility computation as the dominant measured stage at every tested
scale:

- 72 anchors: about 51%;
- 144 anchors: about 54%;
- 288 anchors: about 49%.

The current `FuzzyPredictiveField` constructor already stores every anchor as
a unit vector:

$$
a_i\leftarrow\frac{a_i}{\lVert a_i\rVert}.
$$

The generic perceptual helper nevertheless normalizes every anchor again for
every observation.

For field-owned anchors this is redundant.

## Exact fast path

Keep the generic standalone helper unchanged for arbitrary unnormalized anchor
inputs.

Inside `FuzzyPredictiveField.possibility`, normalize only the incoming
observation:

$$
\hat{x}=\frac{x}{\lVert x\rVert},
$$

then evaluate

$$
d_i=\max(0,1-a_i^\top\hat{x})
$$

directly against the constructor-normalized anchor matrix.

No approximation, truncation, changed metric, or new hyperparameter is allowed.

## Differential reference

The official campaign constructs paired controllers:

- **legacy** — a subclass whose possibility method calls the generic helper and
  therefore renormalizes stored anchors every observation;
- **fast** — the optimized `FuzzyPredictiveField`.

Both controllers receive:

- identical anchors;
- identical observation/outcome trace;
- identical adaptive-resolution parameters;
- identical initial state.

Execution order alternates per step to reduce systematic timing-order bias.

## Exactness metrics

At every step compare:

- perceptual responsibility vector;
- candidate predictions;
- local prediction;
- current local-resolution indices;
- accumulated outcome evidence.

Record:

- maximum responsibility absolute error;
- maximum candidate-prediction absolute error;
- maximum local-prediction absolute error;
- maximum outcome-count absolute error;
- number of local-resolution structural disagreements.

## Performance metrics

Measure in the same process:

1. focused perceptual responsibility time;
2. complete fused prepare+commit time.

Report paired speedups:

$$
S_{perceptual}
=
\frac{T_{legacy,perceptual}}
     {T_{fast,perceptual}},
$$

and

$$
S_{step}
=
\frac{T_{legacy,step}}
     {T_{fast,step}}.
$$

Anchor counts:

$$
72,\quad144,\quad288.
$$

Three deterministic seeds are used.

## Stage-attribution rerun

The existing `fused-stage-attribution-v1` profiler is rerun on the optimized
field at all three anchor scales.

This determines whether perceptual responsibility computation remains the
dominant stage after the exact fast path.

No later optimization target is chosen in advance.

## Frozen gates

The milestone passes only if all conditions hold.

### Numerical / structural parity

1. maximum responsibility error <= **1e-12**;
2. maximum candidate-prediction error <= **1e-12**;
3. maximum local-prediction error <= **1e-12**;
4. maximum outcome-count error <= **1e-12**;
5. local-resolution structural disagreements = **0**.

### Performance at 288 anchors

6. median focused perceptual speedup >= **2.0x**;
7. median complete fused-step speedup >= **1.15x**.

### Re-profile

8. stage-attribution accounting fraction >= **0.90** for every case;
9. at 288 anchors, optimized perceptual share <= **0.30**.

Gate 9 is a preregistered bottleneck-shift hypothesis. If it fails while exact
parity and direct speedup pass, preserve the negative result rather than
weakening the threshold.

## Claim boundary

A passing result supports an exact implementation optimization of the current
research field path.

It does not change Cortex semantics and does not imply a universal runtime
speedup.

## Planned files

- modified `research/fuzzy_predictive_field.py`;
- `research/exact_perceptual_fast_path_campaign.py`;
- `research/exact_perceptual_fast_path_gate.py`;
- `tests/test_exact_perceptual_fast_path.py`.

The official campaign should run only after exactness preflight is green.


## Official outcome

The official paired campaign passed every frozen gate.

At 288 anchors:

- focused perceptual speedup: **48.26×**;
- complete fused-step speedup: **1.96×**;
- optimized perceptual stage share: **2.20%**;
- structural disagreements: **0**.

Across all official cases, maximum numerical discrepancies remained at
floating-point scale; the largest measured difference was approximately
**2.0e-14** in accumulated outcome counts.

The new measured bottleneck is **cache refresh**, at about **45.5%** of the
288-anchor profiled step.

See [EXACT_PERCEPTUAL_FAST_PATH_V1_14_RESULT.md](EXACT_PERCEPTUAL_FAST_PATH_V1_14_RESULT.md) and
[`benchmarks/results/exact_perceptual_fast_path_v1_14_summary.json`](../benchmarks/results/exact_perceptual_fast_path_v1_14_summary.json).

The frozen gates above were not weakened after evaluation.
