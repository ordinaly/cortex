# SL-Bench v0.2 — learned multi-step and action-conditioned profiles

Status: implemented research extension; bounded synthetic characterization.
The six v0.1 GitHub CI runs passed before this extension. No native-runtime
behavior is changed, and the v0.1 folding decision remains in force.

## Question and contract

Can a bounded learner recover the finite distinctions that the exact oracle
exposes when one-step outputs agree but joint future outputs differ?

`cortex.TracePredictiveMemory` learns categorical distributions over output words
for a fixed, finite domain of open-loop action probes. A trial starts from an
observable reset context, receives a declared action word, and returns an output
word. Source states, initial-state mixtures and transition probabilities remain
in evaluator code. The learner does not infer reset contexts from raw history.

Conventions match the oracle: emit at the current state, then transition. The
last action has no effect on this trial's outputs; the fixed domain intentionally
keeps it for compatibility rather than claiming all probe coordinates are
necessary. This is an opportunity for future domain reduction.

One trial updates every observed prefix. Prefix observations from the same trial
are dependent and are not counted as independent trials in the experiment.
Completed trial output is never used for its own prediction. The campaign fits
on 4,096 independent reset trials, freezes the learner, then predicts 1,024 new
trials generated with a separate RNG. No held-out update or fitting occurs.

## Implementation

- Fixed action/output alphabets and H <= 5; explicit domain-size cap.
- Bounded per-context/probe deques and categorical counts; Laplace/Jeffreys-style
  per-category 1/2 smoothing (a declared estimator, not a theoretical primitive).
- LRU eviction of contexts and bounded per-probe recent windows.
- Complete-link empirical partition proposals with minimum support for **every**
  declared probe. Missing evidence leaves a context unresolved and unmerged.
- Exact evaluator audit of proposed partitions against rational source profiles.
- Predictions remain unpooled. A valid empirical fold does not automatically
  authorize changing production predictions or deleting retained evidence.

Storage grows with contexts, probes, trace length and window, but not elapsed
trial count. The action/output domain grows exponentially with H; caps are
intentional. Counts are ordinary categorical estimation, not a novel learning
algorithm or proof of general Sigma–Lambda discovery.

```python
from cortex import TracePredictiveMemory

memory = TracePredictiveMemory(actions=(0, 1), symbols=(0, 1), horizon=3)
probabilities = memory.predict("observed-context", (1, 0, 0))
# Score probabilities before revealing this trial's result.
memory.observe("observed-context", (1, 0, 0), (0, 1, 1))
partition, unresolved = memory.empirical_partition(epsilon=0.25)
```

Output order is Cartesian lexicographic order under the supplied alphabet, exposed
by `memory.words[h]`. Invalid trials fail before mutating retained evidence.

## Fresh world families

Each source has eight hidden states and four observed reset contexts. Exact
context profiles are rational mixtures of state-conditioned oracle profiles.
All action words of lengths 1..3 are in the audit domain.

1. **Correlation:** current-output marginals coincide, but action-dependent
   three-step correlations distinguish two pairs of reset contexts.
2. **Delayed action:** the first two outputs carry no context information;
   the third reveals an action-dependent distinction, yielding four cells.
3. **Noisy correlation:** 10% independent emission corruption challenges the
   same correlation distinction without removing it.
4. **Null:** independent fair emissions erase all context/action information;
   the correct predictive quotient has one cell.

These are fresh families relative to v0.1, with seeds 200..207 and holdout seeds
100200..100207. They are deliberately simple, finite falsification experiments,
not a benchmark of unrestricted reasoning or real-world generalization.

## Baselines and cost controls

- `joint_action`: categorical joint profiles conditioned on action words;
  window 256 per row.
- `joint_blind`: identical categorical estimator, pooling action words by length;
  window 256. Tests the value and expense of action conditioning.
- `factorized_action`: product of position marginals of the same learned joint
  profile. Tests temporal dependence while holding fitted evidence fixed. It
  reuses the joint estimator, so its reported training storage/time is **not**
  the cost of an optimized factorized implementation.
- `joint_action_budget`: same joint learner, window 48. Maximum retained prefix
  traces 2,688, below the action-blind baseline's 3,072. This smaller-budget
  diagnostic was added after viewing the first run; treat it as exploratory,
  not preregistered confirmation.
- `uniform`: fixed uniform distribution, a useful null control with no learning.
- `oracle`: true distribution, evaluator-only.

Compare prediction log loss and true-profile total variation as well as storage.
Successful distinctions are not by themselves proof that their cost is warranted
on sources without those distinctions. Wall times are descriptive single-host
measurements, not a rigorous speed comparison.

## Validation and reproduction

```bash
PYTHONPATH=python:. python -m unittest discover -s tests -p 'test_sl_*.py' -v
PYTHONPATH=python:. python -m research.sl_bench.run_traces --output benchmarks/results/sl_bench_traces_v0_2.json
```

21 tests pass: the 15 v0.1 tests and six additional checks for hidden correlation,
third-step separation, normalization/window bounds, atomic input validation,
context eviction and empirical joint distinctions. Exact probability machinery
retains its independent path-summation tests. Source hashes are recorded before
and after the run and checked for equality. A legacy v1 manifest records the
run and output checksum; it does not enforce OS process limits.

## Boundary and next gate

This extension closes the previous learner's *representation* gap for bounded
multi-step experimental profiles. It does not establish dynamically sufficient
latent-state discovery, arbitrarily long-horizon prediction, or general causal
identification. The controlled actions support conclusions within these source
models only. There is no new real-stream validation: v0.1's sunspot replay did not
test this API and must not be presented as evidence for it.

The next gate is a controller choosing when added horizon/action detail improves
held-out predictions enough to pay its memory cost. The null control makes clear
why always using the richest profile is not the desired final architecture.
