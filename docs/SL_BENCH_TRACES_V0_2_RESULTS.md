# SL-Bench v0.2 — joint-trace results

**Result:** action-conditioned joint profiles recover distinctions missed by one-step or factorized predictions in these finite controlled sources. The null control favors simpler models. No compression or universal learner promotion follows.

32 source/seed runs; each uses 4,096 training reset trials and 1,024 independent held-out trials. That is 131,072 training trials and 32,768 held-out trials per fitted configuration. Models are frozen on holdout. Lower log loss is better.

| Family | Joint + actions | Blind to actions | Joint, small window | Factorized + actions | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| correlation | 0.7203 | 1.0433 | 0.7635 | 2.0911 | 0.6931 |
| delayed_action | 1.4123 | 1.7404 | 1.4581 | 1.4078 | 1.3863 |
| noisy_correlation | 1.6051 | 1.7226 | 1.6412 | 2.0907 | 1.5764 |
| null | 2.1035 | 2.0933 | 2.1472 | 2.0878 | 2.0794 |

The full joint model’s empirical partitions passed the exact epsilon=0.25 oracle in **32/32** runs and matched the minimum feasible cell count in **32/32**. These checks use the exact finite source distributions after training; they are not population guarantees supplied by the learner. The small-window configuration was not used for these partition-success counts.

The factorized model is slightly better on delayed-action outcomes because those outputs have no residual temporal dependence once context/action are known. The uniform no-learning predictor is optimal in expectation on the null world. These controls argue for selective articulation rather than always using the richest representation.

## Retained cost

CPython object-graph estimates include model-owned containers, words, counters and queues, deduplicating shared object identities. They are end-of-fit retained size, not RSS, peak memory, or an implementation-independent byte count.

| Family | Joint KiB | Action-blind KiB | Small-window joint KiB |
| --- | ---: | ---: | ---: |
| correlation | 722.3 | 212.1 | 254.7 |
| delayed_action | 723.0 | 213.8 | 260.6 |
| noisy_correlation | 727.3 | 214.8 | 270.7 |
| null | 726.9 | 214.6 | 272.4 |

The small-window learner retains 2,688 prefix-trace records versus 3,072 for action-blind prediction, but record count alone is not a memory saving: joint profiles need more rows and categories. The byte estimates above make that overhead explicit. Small-window performance remains informative on signal-bearing families; its increased null loss exposes estimation cost. This additional budget ablation was exploratory, added after the initial results, not a preregistered confirmation.

## Decision

Retain the joint action-conditioned representation as an **opt-in research capability**, with finite domain and memory caps. Keep unpooled predictions and no-fold defaults. The next research target is a held-out, cost-aware selector for horizon and dependence structure, including a simple fallback on sources without signal. No universal winner was found.

21 local tests pass. The preceding v0.1 commit passed all six GitHub CI runs; CI for this extension is tracked separately. The result manifest passed validation and verifies unchanged source hashes before/after execution. No new real-data result or native-runtime integration is claimed.

[Contract](SL_BENCH_TRACES_V0_2.md) · [Raw results](../benchmarks/results/sl_bench_traces_v0_2.json) · [Manifest](../benchmarks/results/sl_bench_traces_v0_2.manifest.json)
