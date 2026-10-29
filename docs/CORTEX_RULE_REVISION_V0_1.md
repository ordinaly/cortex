# Counterexample-driven rule revision v0.1

## Contract

`research/rule_revision.py` adds a bounded online monitor around the single-stratum
positive predictive rules selected by `pattern_discovery.py`. Every episode is
predicted and scored before its labels can affect the portfolio. Explicit negative
observations can contradict positive predictions; missing labels remain unknown.

For each rule, the monitor constructs its own proof from the episode's premises.
A contradiction records the episode, rule, predicted and observed atom, and source
fact IDs from one supporting proof. Every rule supporting a contradicted query is
monitored independently. This attributes an unsupported prediction to its rules;
it does not establish whether a sensor, premise, hidden context or changing world
caused the mismatch. The caller must retain the source observations.

## Suspension and repair

The default suspends a rule on one counterexample. An optional policy requires two
counterexample episodes within its last four applicable labeled episodes. Each
episode contributes at most one strike per rule, even if several queries fail.
A mixed episode containing both support and contradiction counts as a strike.
Missing labels and episodes where a rule makes no prediction do not age evidence.
The window is a declared heuristic, not a statistical confidence bound.

Suspended rules move to quarantine; other rules remain active. Repair is grouped
by the affected target symbol. Four subsequently available labeled episodes,
including the triggering episode, propose candidate rules. The next four labeled
episodes validate those frozen proposals with the existing loss/complexity gate.
Proposal and validation IDs are disjoint. Admission occurs after scoring the last
validation episode, so it cannot improve that episode's recorded prediction.
An unsuccessful repair starts another disjoint cycle. The original rule can be
revalidated; its original contradiction remains recorded in the audit history.

Repair can learn an additional observed premise, such as a context marker, to
narrow a rule's scope. This is ordinary relational specialization. It does not
invent hidden context variables, retrieve erased details, diagnose drift, or learn
negative heads. New rules remain in one predictive stratum with retained rules.
A successful repair can leave uncovered contexts where the system abstains.
No error signal arises merely from abstention, so broadening a rule later requires
an additional exploration mechanism not implemented here.

## Bounds, evidence and costs

The default run bound is 1,000 new episodes. Evidence IDs, counters and audit
records are bounded by this run; each pending repair retains at most two blocks.
The existing discovery (10,000 examinations), grounding and proof limits apply.
The whole step executes transactionally on a copy: a budget or validation error
leaves the prior portfolio and evidence IDs unchanged. Copying the bounded state
is an explicit implementation cost; this is not optimized unbounded streaming.

The monitor rejects reused training, validation and stream episode IDs. It cannot
recognize the same observations relabeled with new identifiers. Provenance quality
and label reliability remain caller obligations. External production inference
models are not mutated: this research controller builds a fresh model for each
prediction and needs explicit integration before driving native Cortex memory.

## Evaluation protocol

Run `python research/rule_revision_campaign.py`. The checked-in result is
`research/data/rule_revision_v0_1.json`. Three seeds vary entity names, fact order
and disconnected nuisance predicates. They do not create three independent
families of underlying laws. Each run contains 48 episodes with two observed
queries: one potentially changing target and one stable target. All methods have
the same initial discovery/validation and receive the same labels after scoring.

Four conditions are tested: stationary, context-dependent narrowing after episode
8, complete positive-to-negative reversal after episode 8, and one mislabeled
episode. The four methods are frozen rules, periodic relearning every eight
episodes (four proposal then four validation), immediate targeted suspension, and
two-counterexample targeted suspension. Periodic relearning rebuilds the entire
portfolio; targeted revision preserves unaffected rules.

Reported loss is `2 * wrong + abstained`. Search counts and examined subsets/pairs
measure mining work only; proof evaluation, selection, copying, storage and wall
time are not included. No overall speedup is inferred. Initial training cost is
common and excluded from post-initialization search counts. Scheduled searches
after the final scored episode are included consistently in the counters.

## Observed results

Mean loss over the three seeds (96 scored queries per run):

| Condition | Frozen | Periodic | Immediate revision | Two-counterexample revision |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 0 | 0 | 0 | 0 |
| Context narrowing | 40 | 24 | 25 | 26 |
| Complete reversal | 80 | 48 | 41 | 42 |
| Isolated label error | 2 | 2 | 9 | 2 |

In context narrowing, immediate revision makes one error versus four for periodic
relearning, but abstains 23 times versus 16. Its total loss is therefore slightly
worse at the declared cost ratio. It performs one mining search versus six.
Under complete reversal, immediate revision reduces the frozen model's 40 errors
to one, and the two-counterexample policy to two; most remaining target queries
are abstentions, not newly correct negative predictions. The stable query remains
correct throughout. In the isolated-noise case, the two-counterexample policy
avoids suspension and seven unnecessary abstentions. Neither policy dominates
across these conditions. The result motivates explicit noise/error-cost policies,
not promotion of a universally optimal suspension threshold.

## Limits and next experiment

This benchmark tests mechanics in deliberately small, exactly labeled symbolic
worlds. Noise tolerance is a policy comparison on an isolated-error fixture, not
a calibrated noise model. Hidden context discovery, evidence retrieval, sensor
reliability, learned suspension thresholds and real stream evaluation remain open.
In a complete reversal the learner can stop the positive errors but cannot predict
negative outcomes; abstention is the correct current limitation to expose.

The next experiment should vary noise rates, change frequency and error/abstention
costs with thresholds fixed before evaluation. It should compare revision against
stronger baselines and measure end-to-end computation, then test whether targeted
retrieval or context refinement can distinguish sensor errors from model errors.
