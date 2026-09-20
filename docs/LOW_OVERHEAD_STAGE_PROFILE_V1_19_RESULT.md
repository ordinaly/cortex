# Cortex v1.19-R — Low-Overhead Fused-Stage Crossover Result

Status: **passed engineering attribution milestone**.

Protocol: **low-overhead-fused-stage-profile-v1.19-R**.

Official GitHub Actions run: **35504250664**.

## Objective

v1.18-R showed that fine-grained within-cache attribution overstated the
end-to-end payoff of kernel micro-optimization. v1.19-R therefore measures
each fused stage in its own fresh replay with only one timed boundary.

## CI and validity

The official campaign and full repository CI both passed.

Across all 15 official cases:

- prediction-checksum discrepancy: **0**;
- all stage timings finite;
- all diagnostic fractions valid;
- exact declared seeds/scales present.

## Official coarse-stage profile

| stage | 72 anchors | 144 anchors | 288 anchors |
|---|---:|---:|---:|
| cache refresh | **34.02%** | **36.15%** | **37.22%** |
| candidate readout | 19.27% | 17.64% | **21.80%** |
| hysteresis/state scan | 19.82% | **23.62%** | 17.60% |
| complexity refresh | 4.45% | 7.87% | **13.41%** |
| tracker update | 8.78% | 6.00% | 2.87% |
| perceptual | 7.93% | 5.43% | 2.70% |
| local readout | 4.49% | 3.87% | 3.40% |
| field update | 0.85% | 0.58% | 0.29% |

At 288 anchors the total fused step was about **2146.38 µs**, with:

- cache refresh: **797.00 µs/step**;
- candidate readout: **467.83 µs/step**;
- hysteresis/state scan: **375.61 µs/step**;
- complexity refresh: **287.32 µs/step**.

The independent-replay stage ratios account for about **99.29%** of the
total-step timing at 288 anchors.

## Development/official agreement

The official 288-anchor ordering closely reproduces development:

- development cache refresh: **37.23%**;
- official cache refresh: **37.22%**;
- development candidate readout: **21.14%**;
- official candidate readout: **21.80%**.

This removes the earlier ambiguity: cache refresh is genuinely the largest
coarse Python/NumPy stage.

## Next engineering candidate

Do not return to kernel-math optimization. The hot-path dependency audit shows
that candidate readout, local readout, and complexity consume cached kernels,
while the full cached distance matrix is not required by those step-time
operations.

The next experiment should therefore test **lazy distance materialization**:

1. continue computing dirty-row squared geometry needed for kernel refresh;
2. stop eagerly writing those rows into the full symmetric `distances` cache;
3. mark the distance view stale;
4. materialize/synchronize the full distance matrix only when the public
   diagnostic distance view is explicitly requested;
5. preserve the existing direct eager cache as the paired oracle.

This targets write traffic without changing predictive kernels or reasoning
semantics.

## Claim boundary

This profile characterizes the current Python/NumPy research path on the
declared synthetic traces. It does not characterize the native Rust runtime.

## Reproducibility

- `research/low_overhead_stage_profile_campaign.py`
- `research/low_overhead_stage_profile_gate.py`
- `tests/test_low_overhead_stage_profile.py`
- `docs/LOW_OVERHEAD_STAGE_PROFILE_V1_19.md`
- `benchmarks/results/low_overhead_stage_profile_v1_19_summary.json`
