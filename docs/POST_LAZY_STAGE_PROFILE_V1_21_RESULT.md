# Cortex v1.21-R — Post-Lazy Low-Overhead Stage Profile Result

Status: **passed engineering attribution milestone**.

Protocol: **post-lazy-low-overhead-stage-profile-v1.21-R**.

Official GitHub Actions run: **35508105093**.

## Objective

v1.20-R promoted lazy distance materialization into the default research cache.
v1.21-R therefore repeated the v1.19 independent-replay stage profiler without
changing its stage boundaries, so the new top-level cost ordering could be
measured after promotion.

## Validity

The untouched official campaign passed every frozen gate:

- promoted lazy cache confirmed as the default fused research cache;
- prediction-checksum discrepancy: **0**;
- all timings finite and non-negative;
- diagnostic fractions valid;
- declared seeds and scales exact.

## Official stage profile

| stage | 72 anchors | 144 anchors | 288 anchors |
|---|---:|---:|---:|
| cache refresh | **32.47%** | **33.50%** | **35.55%** |
| candidate readout | 19.34% | 18.48% | **21.21%** |
| hysteresis/state scan | 20.52% | 22.12% | **19.09%** |
| complexity refresh | 4.50% | 8.24% | **13.94%** |
| tracker update | 8.53% | 6.12% | 2.98% |
| perceptual | 7.97% | 5.54% | 2.77% |
| local readout | 4.48% | 4.17% | 3.51% |
| field update | 0.86% | 0.58% | 0.30% |

At 288 anchors:

- total fused step: **2090.32 µs/step**;
- cache refresh: **744.33 µs/step**;
- candidate readout: **444.77 µs/step**;
- hysteresis/state scan: **401.50 µs/step**;
- complexity refresh: **290.88 µs/step**.

The independent-replay stage ratios account for **99.56%** of the total-step
timing at 288 anchors.

## Comparison with v1.19-R

The pre-v1.20 v1.19 official 288-anchor shares were:

- cache refresh: **37.22%**;
- candidate readout: **21.80%**;
- hysteresis/state scan: **17.60%**;
- complexity refresh: **13.41%**.

Post-v1.20, cache refresh falls to **35.55%** but remains the largest stage.
Candidate readout remains second at **21.21%**. Hysteresis rises to **19.09%**
and is now much closer to candidate readout than before.

The absolute v1.19 and v1.21 microsecond totals come from different seed sets
and separate hosted-runner executions, so they should not be interpreted as a
paired performance measurement. The paired v1.20 campaign remains the valid
source for the **1.025×** fused-step speedup claim.

## Engineering interpretation

The optimization landscape is now less skewed:

```text
cache refresh        35.55%
candidate readout    21.21%
hysteresis scan      19.09%
complexity refresh   13.94%
```

Cache refresh is still clearly first, but no single internal-cache timing
hypothesis should be trusted without coarse end-to-end validation. The earlier
v1.17 attribution replication disagreement and v1.18 kernel-ladder rejection
showed why.

## Recommended next step

The next engineering experiment should be **post-lazy cache write/layout
analysis with a coarse paired payoff test**, not another fine-grained timer-led
micro-optimization.

A candidate is worth pursuing only if it removes actual hot-path writes or
allocations while preserving the exact kernel/readout contract. If no such
candidate produces a clear paired fused-step gain, Python cache optimization
should stop and engineering attention should move to candidate readout or to
freezing the next research specification.

## Claim boundary

This result characterizes the current Python/NumPy research path on the
declared deterministic traces. It does not characterize the stable runtime or
external workloads.

## Reproducibility

- `research/post_lazy_stage_profile_campaign.py`
- `research/post_lazy_stage_profile_gate.py`
- `tests/test_post_lazy_stage_profile.py`
- `docs/POST_LAZY_STAGE_PROFILE_V1_21.md`
- `benchmarks/results/post_lazy_stage_profile_v1_21_summary.json`
