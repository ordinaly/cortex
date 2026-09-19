# Cortex v1.15-R — Cache Refresh Attribution

Status: **frozen attribution protocol / official campaign not yet run**.

Protocol ID: \`cache-refresh-attribution-v1\`.

## Motivation

v1.14-R removed redundant anchor normalization from the field-owned perceptual
path. At 288 anchors, perceptual responsibility computation fell from about
48.9% of the fused step to about 2.2%.

The new dominant measured stage is **cache refresh**, at about 45.5% of the
288-anchor step.

Before optimizing cache refresh, v1.15-R decomposes it internally.

## Scientific / engineering question

Where does cache-refresh time actually go after v1.14-R?

No internal winner is preregistered.

The campaign measures:

1. semantic-feature recomputation;
2. outcome-probability recomputation;
3. dirty-row detection;
4. dirty feature-row write;
5. pairwise distance-row recomputation;
6. distance matrix write-back;
7. kernel-row regeneration;
8. kernel matrix write-back;
9. bookkeeping / unattributed remainder.

## Reference path

The profiled implementation duplicates the current
\`PredictiveKernelCache.refresh()\` transition with timers around its exact
substeps.

It must remain behaviorally equivalent to the uninstrumented cache.

No threshold, approximation, sparse policy, or predictive rule changes in this
milestone.

## Workload

Use the same fused prequential trace family used by v1.13-R and v1.14-R.

Anchor counts:

$$
72,\quad144,\quad288.
$$

Three deterministic seeds per scale.

The controller retains the validated sparse-cache settings:

- feature tolerance 0.0025;
- loss-change tolerance 0.001;
- full scan interval 64;
- complexity refresh interval 32;
- full predictive support.

## Dirty-event stratification

Cache refreshes are divided into:

- **clean events** — no semantic row crosses the feature-drift threshold;
- **dirty events** — one or more rows are refreshed.

Report:

- dirty-event fraction;
- mean dirty-row fraction;
- mean dirty rows per dirty event;
- timing shares across all events;
- timing shares conditioned on dirty events.

This separates fixed refresh overhead from geometry-regeneration cost.

## Semantic parity

A paired uninstrumented controller receives the same stream.

At every step compare:

- candidate predictions;
- local predictions;
- local-resolution indices;
- outcome counts;
- cached semantic features;
- cached outcome probabilities;
- cached predictive distances;
- cached candidate kernels.

## Metrics

Per scale:

- cache refresh microseconds per step;
- each substage microseconds per step;
- each substage share of cache-refresh time;
- accounting fraction;
- dirty-event fraction;
- dirty-row fraction;
- maximum paired numerical discrepancy;
- structural disagreement count.

## Frozen gates

The attribution campaign is valid only if:

1. all timing values are finite and non-negative;
2. minimum cache-refresh accounting fraction >= **0.95**;
3. maximum candidate-prediction error <= **1e-12**;
4. maximum local-prediction error <= **1e-12**;
5. maximum cache feature error <= **1e-12**;
6. maximum probability error <= **1e-12**;
7. maximum distance error <= **1e-12**;
8. maximum kernel error <= **1e-12**;
9. local-resolution structural disagreements = **0**;
10. all three declared anchor scales and all three seeds are present.

There is deliberately **no preregistered substage winner**.

The largest measured semantically separable substage at 288 anchors becomes
the next candidate for exact optimization.

## Claim boundary

A passing result supports internal cost attribution of the current Python/NumPy
research cache-refresh implementation.

It does not itself improve performance and does not imply asymptotic complexity
results.

## Planned files

- \`research/cache_refresh_attribution_campaign.py\`;
- \`research/cache_refresh_attribution_gate.py\`;
- \`tests/test_cache_refresh_attribution.py\`.

The official campaign must not run until parity preflight is green.
