# SL-Bench v0.1 — results and promotion decision

**Decision:** keep bounded predictive statistics and explicit evidence retention; do not promote folding as a default Cortex optimization. The exact oracle is useful immediately as a rejection and regression tool.

15 new invariant tests passed; 16 existing source-tree regression checks passed. The oracle additionally checked all 216 three-state deterministic binary sources for stable-memory minimality. No native Rust test or production deployment is claimed.

## Held-out synthetic campaign

8 seeds × 5 families × 6 methods × 6,000 events = 1.44 million held-out updates. Including four development seeds: 360 runs, 2.16 million updates. Lower log loss is better. Families receive equal weight.

| Method | Mean log loss | Delta vs no folding | Mean active cells | Predict/update µs/event |
| --- | ---: | ---: | ---: | ---: |
| guarded | 0.488513 | +0.000000 | 10.400 | 4.379 |
| empirical_fold | 0.492803 | +0.004291 | 8.644 | 4.491 |
| no_fold | 0.488513 | +0.000000 | 10.400 | 2.864 |
| chain_ablation | 0.501656 | +0.013143 | 8.449 | 4.665 |
| no_reopen | 0.514338 | +0.025825 | 8.471 | 4.453 |
| long_memory | 0.506421 | +0.017908 | 8.631 | 4.478 |

The empirical-fold delta has descriptive paired-seed 95% t interval [0.003936, 0.004645]. These are eight seed-level differences averaged across the chosen families, not a confidence interval over arbitrary worlds. Multiple ablations and the small fixed generator family limit inferential claims.

## Component findings

| Component | Evidence | Decision |
| --- | --- | --- |
| Pairwise within-cell bound | Chain-world unsafe-fold audits: empirical complete-link 2.29%; chaining 83.33% | Retain complete-link for experimental folding |
| Reopening | Stationary loss 0.45198 with reopening versus 0.46598 without; drift/recurrence 0.50152 versus 0.56597 | Retain evidence needed for splitting; do not promise recovery after eviction |
| Recent evidence window | Drift/recurrence loss 0.50152 (128 samples/context) versus 0.57494 (10,000) | Window helps here; duration is a task-dependent parameter |
| Confidence gate | Zero folds and identical predictions to no-fold at this budget, with additional comparisons | Conservative gate is not a demonstrated useful compressor |
| Active-cell compression | Roughly 6→3 cells in stationary/chain/drift worlds, same retained evidence | No physical memory-saving claim |
| Aggressive pooling | Helps stationary estimation slightly but increases overall held-out loss | Disabled in default Cortex adapter |
| Bounded retention | 96-context pressure stream respects 32-context cap; loss remains about 0.61614 | Storage bound works; generalization under pressure remains poor |

These are finite component comparisons. They do not establish a special Sigma–Lambda advantage over conventional online context estimators. The no-fold baseline is itself a simple windowed conditional-frequency learner. Complete-link is greedy and can depend on context encounter order; exact optimality belongs only to the exhaustive oracle.

## Loss by world

| World | No fold | Empirical fold | Chaining | Frozen folds | Long window |
| --- | ---: | ---: | ---: | ---: | ---: |
| stationary | 0.45376 | 0.45198 | 0.45300 | 0.46598 | 0.45080 |
| chain | 0.54752 | 0.56509 | 0.60487 | 0.59431 | 0.56483 |
| drift_recurrence | 0.49585 | 0.50152 | 0.50498 | 0.56597 | 0.57494 |
| rare_witness | 0.32929 | 0.32929 | 0.32929 | 0.32929 | 0.32539 |
| budget_pressure | 0.61614 | 0.61614 | 0.61614 | 0.61614 | 0.61614 |

Unsafe-fold means at least one represented cell exceeds the declared true-probability diameter 0.25 at an audit checkpoint. It is not the proportion of incorrect predictions. Audits occur every 64 events. A within-diameter fold may still harm log loss, demonstrating why geometric tolerance alone is not an adequate promotion objective.

## Real domain: annual sunspot direction

Frozen public-domain NGDC data as distributed by statsmodels, 1700–2008. Two previous direction bits predict the next annual increase. Warm-up precedes 1800; 209 forecasts in 1800–2008 are scored before updating. No hyperparameters were fitted on this test segment. This is chronological archive replay, not a live feed.

| Method | Log loss | Brier | Accuracy |
| --- | ---: | ---: | ---: |
| guarded | 0.562189 | 0.186197 | 75.60% |
| empirical_fold | 0.560327 | 0.185774 | 75.60% |
| no_fold | 0.562189 | 0.186197 | 75.60% |
| long_memory | 0.560307 | 0.185767 | 75.60% |
| expanding_rate | 0.678545 | 0.242725 | 59.33% |
| persistence | 0.641525 | 0.205215 | 75.60% |

The default no-fold adapter improves probability loss over the expanding-rate and fixed-confidence persistence baselines, but has the same 75.60% accuracy as persistence. Empirical folding changes log loss by only -0.00186 and retains the same 290 observations at the end. This single short series does not justify promotion or establish superiority over tuned autoregressive/time-series methods. Statistical IID merge guarantees do not apply to this dependent sequence.

## What the oracle has already caught

The one-step candidate partitions fail the three-step exact audit in delayed-witness, action-blindness and stochastic-aliasing worlds. Each rejection contains a pair of states and a distinguishing action probe. Exact duplicate states can be safely identified within the declared profile domain. This is the intended purpose: make implementation claims fail concretely when they erase consequential distinctions.

## Next gate

Require net retained-storage or prediction benefit under a jointly declared distortion/loss budget on fresh world families before enabling compression. Extend the learner beyond one-step binary profiles so it can respond to the delayed/action witnesses already produced by the oracle. Production native integration remains a separate gate.

Artifacts: [contract](SL_BENCH_V0_1.md), [raw results](../benchmarks/results/sl_bench_v0_1.json), [computation manifest](../benchmarks/results/sl_bench_v0_1.manifest.json).
