# Selective training v0.3

Status: opt-in finite research experiment, 21 October 2026. Extends
[question learning v0.2](CORTEX_QUESTION_LEARNING_V0_2.md); native runtime unchanged.

## Mechanism

Training now observes only the selected property. `PartialEvidence` stores an
unasked answer as `None`; it never updates that property's counts. The learner
receives the cheap context and later outcome on every episode, equally across
policies. Question choices are frozen before the current answers and outcome.

For each context/question pair, the acquisition score is estimated answer
prevalence times its complement times squared conditional outcome-probability
separation, plus 0.1 divided by the square root of the acquired count plus one.
This is a heuristic signal/coverage score, not a calibrated information-gain or
confidence bound. Beta(1,1) predictive means smooth conditional outcome counts.
A 20% random-exploration probability gives each question at least 10% selection
probability per queried episode. This is a probabilistic allowance, not an exact
reserved count or a finite-run discovery guarantee.

After discovery, a single question is proposed using the context-frequency
weighted separation score. The proposal and all parameters freeze before 64
fresh validation episodes acquire that question. Adoption requires average
Brier improvement above 0.02 and at least four discovery observations in each
context/answer cell. No refitting or alternative selection on validation follows
rejection. These empirical thresholds are not statistical error guarantees.

Only acquired marginals are estimated: there is no fabricated joint distribution
of two properties that were never jointly observed. At evaluation, the selected
question becomes a two-alternative `FiniteModel`; `PossibilityFilter` supplies
exact conditioning and prediction. The rejected model uses its context-only
predictive mean. Supplied deterministic sensors and within-episode identity
transitions remain unchanged. No temporal transition learning is claimed.

Adaptive selection depends only on previous evidence and the current context.
The experiment assumes stationary, independently sampled episodes. Its counting
rules do not solve selection bias for arbitrary dependent or drifting streams.

## Equal-budget experiment

Seeds 40–49 are distinct from v0.2. Each policy sees the same contexts and free
outcomes, and spends the same discovery budget (64, 128 or 256 queries):

- Active: one selected query per episode, using updated counts and exploration.
- Random: one uniformly random query per episode.
- Exhaustive: both queries on a uniformly sampled half of episodes. With an odd
  budget, one additional episode acquires a single random property.

All policies then spend 64 validation queries and 512 evaluation queries. During
evaluation every policy acquires its proposed question, even when validation
rejected it and the answer is ignored. This intentionally fixes all acquisition
costs to isolate learning quality. It is not the deployment policy from v0.2,
which can abstain from queries. Total loss is evaluation decision error plus
0.1 times **all** acquisition queries, divided by 512 evaluation episodes.
Training/validation decisions themselves are not scored as task outcomes.

Context and outcome observations are free in this fixture; runtime and memory
costs are not budget-matched. The benchmark uses the existing learnable and
noise-only worlds with swapped relevant-probe identities across seeds. Probes
and representation vocabulary remain supplied. The policy can choose *which*
question, but does not yet learn when to stop training or skip an episode.

## Results

Machine-readable per-seed results and parameters:
`research/data/selective_training_v0_3.json`.

| Discovery budget | Active error | Random error | Exhaustive error | Total queries, each policy |
|---|---:|---:|---:|---:|
| 64 | 12.01% | 13.85% | 18.44% | 640 |
| 128 | 9.98% | 9.98% | 9.98% | 704 |
| 256 | 9.98% | 9.98% | 9.98% | 832 |

At budget 64, adoption occurred in 10/10 active, 8/10 random and 6/10 exhaustive
learnable cases. All adopted in the larger-budget learnable cases. Every policy
rejected refinement on all ten noise-only seeds at every budget. Seeds repeat
across budgets: these are not thirty independent control worlds per policy.

At budget 64, mean total loss was 0.2451 active, 0.2635 random, 0.3094 exhaustive.
Active Brier scores at budgets 64/128/256 were 0.1061/0.0944/0.0916, compared with
0.1176/0.0961/0.0929 random and 0.1290/0.0968/0.0921 exhaustive.

These are modest descriptive results on ten seeds, not proof of general sample
efficiency. At larger budgets the accuracy advantage disappears. No parameters
were changed after inspecting this campaign's outcomes. Multiple hypotheses,
noisy sensors, changing regimes and larger question vocabularies remain untested.

## Validation and next step

```bash
python -m pytest tests/test_selective_training.py tests/test_question_learning.py tests/test_possibility_geometry.py
python research/selective_training_campaign.py --output research/data/selective_training_v0_3.json
```

Tests verify missingness, immutable snapshots, prefix-only question selection,
exploration, exact budgets (including odd and zero budgets), equal free outcomes,
fresh validation, reference conditioning and full acquisition cost accounting.
The manual question-learning workflow also reproduces this campaign.

The next useful test is whether the policy can stop spending on uninformative
questions while retaining enough exploration to notice a newly useful one.
Then introduce unannounced regime changes and recurrence. Current results do
not justify native integration or replacing the established plasticity controller.
