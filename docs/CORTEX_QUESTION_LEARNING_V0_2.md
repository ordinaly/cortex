# Question learning v0.2

Date: 21 October 2026. Status: bounded research experiment; native runtime unchanged.
Depends on [Possibility Geometry v0.1](CORTEX_POSSIBILITY_GEOMETRY_V0_1.md).

## Question and implementation

Can a supplied distinction earn adoption through fresh observations, and can the
resulting probability model choose when a paid observation improves a decision?

`research/question_learning.py` learns outcome probabilities conditional on a
cheap context and either zero or one selected binary probe. It begins with a
context-only representation, fits candidate refinements on exploratory evidence,
proposes one using discovery Brier-score improvement, then validates the frozen
proposal on a separate stream. It adopts only if both gains exceed 0.02 and each
discovery context/answer cell has at least 16 observations. These are empirical
fixture thresholds, not confidence guarantees. A rejected proposal is not
replaced by another proposal selected on the same validation data.

Outcome estimates are Beta(1,1) posterior predictive means. The joint distribution
of the two probe answers uses a Dirichlet(1,1,1,1) predictive mean separately for
each context. The model retains counts, but does not propagate a posterior over
parameters through planning or supply calibrated parameter confidence intervals.
Correlations between probe answers are retained rather than assuming independence.

For a new episode, the learner builds a finite model with four mutually exclusive
answer configurations. Candidate probabilities come from acquired observations.
A query conditions these possibilities through `PossibilityFilter`; the predicted
outcome is also obtained through that exact reference. An unanswered property
remains unresolved. Probe observation rules are supplied deterministic sensors;
temporal transitions are supplied identity transitions within an episode. This
milestone learns outcome laws and question relevance, not transition dynamics.

The query policy computes the expected reduction in binary decision error and
asks the best question only when that reduction exceeds its cost. It asks at most
one question per evaluation episode. Contexts where an answer cannot change the
optimal decision generally do not merit inspection. All probabilities and query
values use rational arithmetic; reporting converts to floats.

This makes one operational-semantic idea executable: a distinction gains a role
through validated consequences, rather than through its label. Probe identities
swap across seeds. It remains finite feature selection: the sensors, contexts,
hypothesis configurations and available representation changes are supplied.

## World and observation boundary

`research/question_learning_campaign.py` supplies a small stochastic environment.
Each episode has a visible binary context, two initially hidden observable bits,
and a binary outcome. One probe is relevant in context 1; context 0 has outcome
probability 0.1 regardless of probe answers. In learnable cases, the relevant bit
changes outcome probability from 0.1 to 0.9 in context 1. In controls, context 1
has probability 0.5 independent of both probes. Irreducible variation remains.

The world alone uses those probabilities. The learner receives evidence records
containing only context, actually acquired answers, and the later outcome. Its
code does not read world parameters or latent simulator fields. `Episode.ask`
charges once per acquired sensor and refuses unknown, duplicate, or post-outcome
questions. This is an API discipline, not a security sandbox. Oracle answers are
perfect observations in this version; noisy sensors and interventions are deferred.

Forecasts are frozen before the outcome is revealed. Discovery, validation and
evaluation IDs cannot overlap; independent seeded world streams provide the
experimental separation (distinct IDs alone do not prove independence). Evaluation
outcomes never update the fitted model. The model can refuse an unseen context;
it does not autonomously repair one.

## Protocol and cost accounting

The frozen run uses seeds 20–29, in both learnable and no-distinction worlds.
Each case has 512 discovery episodes, 512 validation episodes, and 512 evaluation
episodes. Discovery and validation query both probes: 1,024 queries each, or
2,048 acquisition queries per case. Every evaluated policy shares this fitted
model and this acquisition cost. The query cost is 0.1 relative to a wrong
decision cost of 1.

Four evaluation policies are compared:

- Active: choose at most one question by expected decision improvement.
- Random matched: query exactly as many episodes as active, selecting episodes
  and question identities randomly. It receives only the total quota, not active's
  locations, answers, or outcomes. This is an offline budget-matched baseline.
- None: use the learned mixture without acquiring episode-specific answers.
- All: acquire both probe answers; predict using the adopted conditional model.

The reported total loss is decision error plus **evaluation** question cost.
Initial acquisition costs are reported separately and are equal across policies.
Amortizing those costs over just 512 evaluation episodes adds 0.4 to each policy's
reported loss. There is no claim of more efficient training in this experiment.

## Results

Committed machine-readable output: `research/data/question_learning_v0_2.json`.
Means over ten seeds, with 512 evaluation episodes per seed:

| Learnable worlds | Queries per 512 episodes | Decision error | Brier score | Error plus query cost |
|---|---:|---:|---:|---:|
| Active | 260.6 | 10.12% | 0.0913 | 0.1521 |
| Random matched | 260.6 | 25.10% | 0.1525 | 0.3019 |
| None | 0 | 30.10% | 0.1721 | 0.3010 |
| All | 1024 | 10.12% | 0.0916 | 0.3012 |

The relevant distinction was adopted in 10/10 learnable cases and a distinction
was adopted in 0/10 control cases (all ten controls rejected refinement).
Active made no evaluation queries in the controls. Control decision error was
31.04%, compatible with persistent unresolved randomness in this small sample;
no universal calibration or statistical superiority claim follows from ten seeds.

The small Brier difference between active and all is possible because all also
conditions on finite-sample estimates in contexts where active does not inspect.
These comparisons measure acquisition policy under a shared learned model, not
an advantage of topology over an equivalent probability mixture. No simplicial
algorithm or autonomous native Cortex controller participates in this benchmark.

## Reproduce and validate

```bash
python -m pytest tests/test_question_learning.py tests/test_possibility_geometry.py
python research/question_learning_campaign.py --output research/data/question_learning_v0_2.json
```

Tests cover fresh-validation rejection, noise-only rejection, swapped probe
identities, unresolved alternatives, costly-query abstention, oracle access,
immutable forecasts, outcome-blind selection, evaluation budget matching, evidence
reuse rejection and unsupported contexts. PR CI runs the unit tests. A separate
manual workflow reproduces the campaign and uploads its report.

## Next boundary

Before a native integration, test noisy sensors, costly exploration during
learning, changing regimes, and prospective creation of a new distinction after
failure. Add unknown-applicability handling and parameter uncertainty. Expanding
or composing the probe vocabulary is a separate task from selecting a supplied
feature. The world oracle supplies evidence; it does not discover the explanation
for Cortex.
