# Cortex v1.15-R — Cache-Refresh Attribution

Status: **frozen attribution protocol / official campaign passed**.

Protocol ID: **cache-refresh-attribution-v1.15-R**.

## Motivation

v1.14-R made perceptual responsibility computation exact and much faster. At
288 anchors, perceptual share fell to about **2.2%**, while **cache refresh**
became the new dominant measured stage at about **45.5%** of the fused research
step.

The next engineering action is therefore **measurement, not optimization**.

No internal cache substage is assumed to be the winner in advance.

## Question

Where does cache-refresh time actually go after each field update?

The campaign decomposes the current sparse cache refresh into six engineering
groups:

1. semantic-feature recomputation;
2. outcome-probability refresh;
3. dirty-anchor detection;
4. pairwise distance-row recomputation;
5. kernel-row regeneration;
6. writes and bookkeeping.

The profiler also retains finer raw substages so write costs can be separated
from numerical recomputation when interpreting the result.

## Exact reference

The profiled cache is paired against the existing SupportSparsePredictiveCache.

Both controllers receive identical anchors, observations, outcomes, resolutions,
sparse-refresh tolerance, and local-resolution state.

After every step the campaign compares candidate predictions, local prediction,
field outcome counts, cached semantic features, cached outcome probabilities,
cached predictive distances, cached kernels, and local-resolution indices.

The attribution implementation is valid only if it preserves the existing cache
semantics to floating-point precision.

## Attribution groups

### Semantic features

Recompute the square-root semantic feature representation from current outcome evidence.

### Outcome probabilities

Refresh the smoothed outcome-probability matrix used by predictive readout.

### Dirty detection

Measure per-anchor semantic drift against cached feature rows and determine
which anchors exceed the existing feature tolerance.

### Distance rows

For dirty anchors only, recompute predictive distance rows against the complete
anchor set.

### Kernel regeneration

Regenerate candidate-resolution kernel rows from the updated distances.

### Writes and bookkeeping

Aggregate cached feature writes, symmetric distance-row writes, symmetric
kernel-row writes, dirty-mask updates, revision counters, and refresh statistics.

## Scale sweep

The campaign uses 72, 144, and 288 anchors.

Development seeds: **0, 1, 2**.

Reserved untouched official seeds: **100, 101, 102**.

Each case uses the existing deterministic equivalent/divergent/reconverged trace.

## Metrics

For each scale, report the median total attributed refresh microseconds per step,
share and microseconds per step for each engineering group, dirty-event fraction,
mean dirty-row fraction on dirty events, and attribution accounting fraction.

The raw finer substages are also retained.

## Frozen official validity gates

Before official evaluation, the development implementation must satisfy:

1. all cache/prediction numerical discrepancies <= **1e-12**;
2. structural disagreements = **0**;
3. every timing is finite;
4. attributed accounting fraction >= **0.80** for every case;
5. all group shares lie in **[0,1]**;
6. group shares sum to at most **1.0** apart from floating-point tolerance;
7. dirty-event and dirty-row fractions lie in **[0,1]**;
8. exactly the declared seeds and anchor scales are present.

There is deliberately **no gate naming the dominant internal substage**.

The official measurement determines the next optimization target.

## Interpretation rule

The next optimization should target a group only if it is both a large measured
share at the largest tested scale and semantically separable enough to optimize
exactly or behind an explicit approximation contract.

If no group clearly dominates, preserve that result rather than inventing an
optimization target.

## Timing caveat

Fine-grained perf_counter_ns calls add overhead. Within-refresh stage shares,
scale trends, dirty-event duty, and relative dominance are the primary quantities.

Absolute attributed microsecond values should not be treated as directly
comparable with uninstrumented production timings.

## Claim boundary

A passing campaign supports only an engineering attribution claim about the
current Python/NumPy research cache-refresh path.

It does not establish a universal Cortex bottleneck or a production runtime limit.

## Development outcome before freeze

The 9-case development campaign passed every validity gate with **zero**
numerical discrepancies and **zero** structural disagreements.

At 288 anchors, median cache-refresh attribution was:

- distance-row recomputation: **39.73%**;
- kernel regeneration: **35.14%**;
- writes/bookkeeping: **17.87%**;
- semantic features: **2.66%**;
- dirty detection: **1.66%**;
- outcome probabilities: **1.47%**.

The median accounting fraction was **0.9862**.

Every refresh event was dirty under this trace, but only about **13.30%** of
anchor rows were refreshed per event at 288 anchors.

No gate or mechanism may now be changed before the untouched official
evaluation. Any change after opening official seeds requires a new protocol
version and fresh seeds.

## Official outcome

The untouched official campaign on seeds **100, 101, 102** passed every frozen
validity gate.

At 288 anchors:

- distance-row recomputation: **41.10%**;
- kernel regeneration: **35.50%**;
- writes/bookkeeping: **15.48%**;
- semantic features: **2.90%**;
- dirty detection: **1.88%**;
- outcome probabilities: **1.55%**;
- accounting fraction: **98.44%**;
- mean dirty-row fraction: **13.40%**.

All numerical cache and prediction discrepancies were exactly **0** in the
paired campaign, with **0 structural disagreements**.

The frozen gates were not weakened after official evaluation.

See [CACHE_REFRESH_ATTRIBUTION_V1_15_RESULT.md](CACHE_REFRESH_ATTRIBUTION_V1_15_RESULT.md)
and [the committed aggregate summary](../benchmarks/results/cache_refresh_attribution_v1_15_summary.json).
