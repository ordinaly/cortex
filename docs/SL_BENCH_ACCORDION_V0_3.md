# Physical accordion memory v0.3

## Contract

`cortex.AccordionMemory` is an opt-in binary, observed-context research model.
It implements a limited fold/witness/reopen cycle, independently of the native
runtime and the multi-step trace learner. It does not certify Sigma–Lambda
sufficiency or dynamically stable predictive equivalence.

- Each active context owns a bounded byte window (256 observations by default).
- Every 256 global updates, full windows outside cooldown are grouped by
  complete-link empirical mean distance (at most 0.10 for every pair).
- A candidate replaces full windows with one bounded pooled window plus the
  last 32 observations per member. The pool interleaves equal-depth suffixes;
  it is not a chronological global window or an exact sufficient statistic.
- Commit the fold only if its complete retained object-graph estimate drops
  by more than 5%, including metadata and counters. Otherwise roll back the
  original dictionaries and counters. This is a local acceptance condition,
  not a guarantee about subsequent memory, peak allocations, or process RSS.
- A member reopens when its witness mean differs from the pool before that
  update by more than 0.30. Only retained witnesses seed its restored window.
  Discarded history cannot be reconstructed. A full-window cooldown delays
  refolding. Remaining members rebuild their pool from surviving witnesses;
  a singleton cell dissolves into its witness window.
- LRU eviction bounds contexts and represents explicit information loss.

The estimator uses `(ones + 0.5)/(observations + 1)`. Decisions and thresholds
are empirical, with no confidence or population-preservation certificate.
The estimator counts Python instance-owned objects, not shared class machinery
or allocator bookkeeping. Candidate construction temporarily retains old and
new structures; that peak cost is not represented by the retained-size gate.

## Bounded campaign

```bash
PYTHONPATH=python:. python -m unittest discover -s tests -p 'test_sl_*.py' -v
PYTHONPATH=python:. python -m research.sl_bench.run_accordion --output benchmarks/results/sl_bench_accordion_v0_3.json
```

Five families, eight paired seeds (300–307), three configurations, 24,000 events
per run: 120 model runs / 2,880,000 updates. Configurations are unfurled,
accordion with reopening, and frozen folds. All use identical context and
window budgets; these are mechanism controls, not optimized competing models.

The families are redundant low/high Bernoulli contexts, independent fair noise,
a graded similarity chain, a single drifting context, and context pressure
(96 contexts under a cap of 32). Other families have 16 observed contexts.
The local change occurs at event 12,000. Prediction always precedes training.
Warmup is 8,192 events; retained bytes are sampled every 1,024 events thereafter.
Update timing includes prediction, folding comparisons, candidate construction,
and the memory gate; measurement scans themselves are excluded.

Recovery is five consecutive *visits to the drifting context* with predictions
within 0.15 of its new true probability. Delay is measured in global events.
Report recovery rate as well as conditional delay; unrecovered runs are censored.

The saved source hashes bind the executed code. The local repository is a
partial snapshot with a local snapshot commit, not a full clone of PR #96;
the manifest records that actual local commit and dirty state. The PR parent
at publication is `a452d5981150330031b101fb34de901adf0d5c94`. No hard process caps
were enforced. Seeds were reused from the diagnostic ungated run when repairing
the gate; these are development measurements, not untouched external validation.

## Decision

See [results](SL_BENCH_ACCORDION_V0_3_RESULTS.md). Keep physical folding
experimental. The existing predictive adapter still disables folding by default.
No native memory, entity lifecycle, or closed-loop behavior changes here.

The next discriminating experiment is a cost-aware fold scheduler: reduce
repeated comparisons of rejected candidates, then freeze its settings and test
on new streams. A short-window/change-detector baseline is also needed to
separate the benefit of fast forgetting from the benefit of shared storage.
Multi-step, action-sensitive folding and a real streaming test remain untested.
