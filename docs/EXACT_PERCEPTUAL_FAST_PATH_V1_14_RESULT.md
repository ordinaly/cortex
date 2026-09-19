# Cortex v1.14-R — Exact Perceptual Fast Path Result

Status: **passed research optimization milestone**.

Protocol: \`exact-perceptual-fast-path-v1.14-R\`.

Official run: GitHub Actions run \`35436434761\`.

## Objective

v1.13-R showed that perceptual responsibility computation consumed roughly
half of fused research-step time.

The field constructor already normalized and stored every anchor, but the
generic perceptual helper normalized those anchors again for every observation.

v1.14-R replaces that field-owned path with:

$$
\hat{x}=\frac{x}{\lVert x\rVert},
$$

$$
d_i=\max(0,1-a_i^\top\hat{x}),
$$

using the already normalized anchor matrix.

The generic standalone helper remains unchanged.

## Exactness result

Paired legacy and optimized controllers received identical anchors, traces and
adaptive-resolution state.

Measured maximum discrepancies across all official cases:

| quantity | maximum absolute error |
|---|---:|
| perceptual responsibilities | **2.78e-15** |
| candidate predictions | **3.33e-16** |
| local predictions | **2.22e-16** |
| outcome counts | **2.00e-14** |

Local-resolution structural disagreements:

$$
\boxed{0}
$$

The small numerical differences arise from floating-point re-normalization
order and are far below the frozen \(10^{-12}\) gates.

## Paired speedup

| anchors | perceptual speedup | complete fused-step speedup |
|---:|---:|---:|
| 72 | **15.59×** | **1.83×** |
| 144 | **27.20×** | **2.01×** |
| 288 | **48.26×** | **1.96×** |

The complete paired controller therefore nearly doubled throughput at the
largest tested scale without changing structural behavior.

## Stage-attribution rerun

The original v1.13-R perceptual share was approximately:

- 72 anchors: 51.0%;
- 144: 54.0%;
- 288: 48.9%.

After v1.14-R:

| stage | 72 | 144 | 288 |
|---|---:|---:|---:|
| cache refresh | **37.1%** | **43.3%** | **45.5%** |
| candidate readout | 17.6% | 15.9% | 21.0% |
| hysteresis scan | 18.7% | 18.5% | 12.7% |
| complexity telemetry | 4.0% | 6.5% | 12.1% |
| perceptual | **7.3%** | **4.8%** | **2.2%** |

Perceptual responsibility computation is no longer the bottleneck.

At 288 anchors, its median profiled cost fell to about **43.6 µs/step**.

## New measured bottleneck

The next dominant stage is unambiguous:

$$
\boxed{\text{cache refresh}}
$$

at roughly **45.5%** of measured time at 288 anchors.

This becomes the B2 optimization target, subject to a fresh decomposition of
what cache refresh is actually doing before any implementation change is
chosen.

Candidate readout is the second-largest stage at roughly 21%.

## Interpretation

This result validates the optimization strategy used throughout the recent
Cortex engineering line:

1. measure;
2. optimize an exact redundant operation;
3. preserve semantics;
4. re-profile;
5. let the new measurement choose the next target.

The result is workload-specific and applies to the Python/NumPy research path.
It is not a universal Cortex speedup claim.

## Reproducibility

Implementation:

- modified \`research/fuzzy_predictive_field.py\`;
- \`research/exact_perceptual_fast_path_campaign.py\`;
- \`research/exact_perceptual_fast_path_gate.py\`;
- \`tests/test_exact_perceptual_fast_path.py\`.

Frozen protocol:

- [\`EXACT_PERCEPTUAL_FAST_PATH_V1_14.md\`](EXACT_PERCEPTUAL_FAST_PATH_V1_14.md)

Aggregate result:

- [\`benchmarks/results/exact_perceptual_fast_path_v1_14_summary.json\`](../benchmarks/results/exact_perceptual_fast_path_v1_14_summary.json)

## Next engineering milestone

B2 should decompose **cache refresh** before optimizing it.

The next campaign should attribute refresh cost among:

- semantic feature recomputation;
- outcome-probability refresh;
- dirty-anchor detection;
- pairwise distance updates;
- kernel-row regeneration;
- bookkeeping / scan logic.

No optimization target inside cache refresh should be assumed before that
decomposition is measured.
