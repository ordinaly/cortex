# Physical accordion v0.3 results

The complete-size gate produces measured retained-memory savings on three
families. Reopening helps local drift, but computation still costs more and
noise-only predictions degrade slightly. This is a bounded development result,
not a general performance or theoretical sufficiency claim.

## Paired results

Means over eight seeds. Loss is post-warmup natural-log loss (lower is better).
Memory is estimated retained Python object bytes, including witnesses and indices.

| Family | Unfurled bytes | Accordion bytes | Reduction | Unfurled loss | Accordion loss | Update cost ratio |
|---|---:|---:|---:|---:|---:|---:|
| redundant | 10658 | 9152 | 14.1% | 0.326914 | 0.326914 | 2.07× |
| fair | 10528 | 9000 | 14.5% | 0.695198 | 0.695655 | 1.97× |
| chain | 10527 | 10499 | 0.3% | 0.578176 | 0.578176 | 3.61× |
| local_drift | 10527 | 9158 | 13.0% | 0.335396 | 0.331171 | 2.02× |
| pressure | 12280 | 12252 | 0.2% | 0.615687 | 0.615687 | 1.02× |

The tiny byte differences in chain/pressure are object-accounting differences,
not compression: both make zero folds. The gate rejects the chain's candidates;
comparison and rollback costs remain. Under pressure, contexts are evicted before
windows fill, so folding does not address that bottleneck.

## Drift and reopening

| Method | Recovered runs | Mean global events to recovery | Post-warmup loss |
|---|---:|---:|---:|
| unfolded | 8/8 | 3410.8 | 0.335396 |
| accordion | 8/8 | 1831.4 | 0.331171 |
| frozen | 2/8 | 3384.0 | 0.378637 |

Accordion's mean recovery delay is 46.3% lower than unfurled memory. Frozen-fold
delay is conditional on its two successes and cannot be compared as if all eight
runs recovered. Mean accordion loss improves on local drift, but not every seed.

## Variation across seeds

Paired accordion-minus-unfurled post-warmup loss differences. These ranges are
observed seed variation, not confidence intervals or significance claims.

| Family | Mean difference | Minimum | Maximum |
|---|---:|---:|---:|
| redundant | -0.000000 | -0.000468 | +0.000487 |
| fair | +0.000457 | -0.000440 | +0.002521 |
| chain | +0.000000 | +0.000000 | +0.000000 |
| local_drift | -0.004225 | -0.006386 | +0.000427 |
| pressure | +0.000000 | +0.000000 | +0.000000 |

## Validation and limits

- 27 SL-Bench tests pass, including physical deletion, full-size savings,
  exact rollback, witness-only reopening, complete-link grouping and eviction.
- 16 existing closed-loop/regime regression functions pass via direct invocation
  (`pytest` is unavailable locally). Native extension/full-suite validation is
  deferred to CI.
- The computation manifest validates with source/output hashes against this
  workspace. It uses the legacy v1 schema, which lacks enforced resource caps.
- 120 model runs / 2.88 million updates; predictions precede updates.
- Timing is one ordered run in this environment, not a hardware benchmark.
- Object-graph bytes are neither process RSS nor peak allocation. Small
  differences can arise from Python sharing and dictionary allocation.
- The short pooled window can forget faster even without reopening. No claim
  that reopening alone explains all recovery gains is justified.
- Empirical binary similarity does not establish preservation of delayed,
  action-dependent, causal, or relational distinctions. No exact-oracle
  certificate is claimed for these stochastic folds.

The accordion cycle now earns a **storage** benefit in this tested regime.
It does not yet earn default promotion on combined accuracy and compute cost.
The next test should target rejected-candidate overhead and include a fast
forgetting baseline with frozen settings on new seeds.

[Contract and command](SL_BENCH_ACCORDION_V0_3.md).
Raw data: `benchmarks/results/sl_bench_accordion_v0_3.json`; provenance is in
the adjacent `.manifest.json`.
