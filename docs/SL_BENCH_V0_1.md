# Finite Sigma–Lambda oracle and SL-Bench v0.1

Status: **implemented, finite tests passed, pilot ablations completed**.
This is a falsifiable executable subset of Sigma–Lambda, not an oracle for the
whole theory. The Cortex integration is opt-in Python research code; native
Rust allocation, the frozen spec, and the existing closed loop are unchanged.

## Authoritative scope

Read on 2026-10-09:

- `sigma_lambda_revised_2026.tex`, canonical whitepaper dated October 2026:
  finite partiality, non-identification, grounded observations, refinement,
  Lambda-down compression and Lambda-up differentiation. This supersedes the
  earlier mandatory inexhaustibility axiom.
- `Sigma_Lambda_Theory_Audit_2026-09-23.md`: distinguishes predictive quotients
  from stable refinements of supplied coarse partitions; requires coherent
  approximate partitions and separates one-step from long-horizon distortion.
- `Sigma_Lambda_Core_XVIII_Representation_Neutral_Predictive_World_v0_3.md`:
  **proposed, not frozen**. Used as the explicit experimental interface for
  probe-relative prediction, retained evidence and reopenability, not promoted
  to a theorem by this implementation.
- Current remembered articulation-profile/approximate-role milestone supplies
  the optional typed-relation permutation test. Equal weighting of relation
  types is a declared experimental choice, not a canonical scalar metric.

Repository base: `01f0fdeb1cf7f6ac353c3eb7894146fa1fd96e65`.
PRs #92 and #95 were open at inspection; neither is merged or altered here.
The local working copy was assembled from commit-pinned connector reads because
ordinary authenticated cloning was unavailable. It is a selected source snapshot,
not a full local checkout of repository history. Changes are submitted against
the actual upstream tree.

## Six-step implementation status

| Step | Delivered in v0.1 | Boundary |
| --- | --- | --- |
| Finite oracle | Exact rational trace profiles, predictive quotients, stable memory, exhaustive partition frontier, counterexample certificates, typed semantic-role distances | Finite controlled Markov sources; no whole-theory/physics oracle |
| SL-Bench | Five exact worlds and five stochastic stream families | Observed context labels supplied; no arbitrary feature/entity discovery |
| Approximate learner | Bounded per-context/probe statistics, coherent greedy folding, uncertainty gate, reopening, LRU eviction | Binary outcomes, fixed probe grammar; no convergence theorem |
| Ablations | Six configurations, 4 development and 8 disjoint held-out seeds, 6,000 events/run | Pilot family-level characterization; not broad generalization |
| Cortex integration | Public `PredictiveMemory`, policy and `PredictiveEvidenceAdapter`; `EvidenceAtom` ingestion | Research interface; default adapter disables folding following ablations |
| Real stream | Chronological sunspot archive replay, 209 scored annual forecasts | Historical replay, not live ingestion; one small domain |

## Exact computational contract

`research/sl_bench/oracle.py` consumes a finite state carrier, rational stochastic
kernels `K[action][state][next_state]`, and rational emissions
`E[state][symbol]`. Each step emits at its current state and then transitions.
An open-loop action word of length h induces a distribution over h-symbol words.
The final action is observationally redundant under this emission convention;
including it keeps the probe type uniform.

The declared profile includes all action words and output words at lengths 1..H.
The exact predictive quotient identifies identical profiles. Total-variation
distance is maximized over probes. Approximation uses actual partitions with
maximum **within-cell pairwise** distance bounded by epsilon. It never forms a
quotient by taking the transitive closure of tolerance-neighbor edges.

`frontier` enumerates every set partition and minimizes diameter at each cell
count. This is an exact finite cost/distortion frontier for the declared cost
(number of cells), not a claim that cell count is the unique theoretical cost.
`audit_partition` returns a pass/fail result, distortion, optimum feasible cell
count and explicit pair/probe counterexamples. A pass certifies only the supplied
profile domain. It does not certify predictive quality of a separately fitted
representative or dynamics outside that domain.

`stable_memory` iteratively refines a supplied partition until emissions agree
and transition mass to every block agrees for every action. This is an
output-respecting strongly lumpable refinement. It is deliberately separate from
finite-H trace equivalence. A supplied initial partition may retain additional
structural distinctions.

`semantic_role_distances` exhaustively minimizes maximum typed normalized
relation distortion over bijections mapping x to y. It operates on a supplied
relational carrier, not inferred physical entities. It is not part of the
streaming learner or the component performance claims.

Caps: at most 8 states for partition enumeration; at most 7 for permutations;
H <= 5; profile entry budget <= 200,000. Invalid distributions, float inputs to
the exact world, malformed partitions and excess resources raise explicit
exceptions. The campaign uses at most 4 states and H=3. No resource truncation
silently produces a pass.

## Adversarial coverage

| World | Failure exposed |
| --- | --- |
| Delayed witness | Equal current outputs conceal distinct later outputs |
| Action blindness | Passive-only probes miss an intervention distinction |
| Nuisance duplicates | Distinct labels can be predictively redundant |
| Non-transitive chain | Adjacent tolerance matches cannot justify one cell |
| Stochastic aliasing | Current emissions conceal transition-law differences |
| Stationary binary stream | Pooling can reduce estimation noise |
| Chain stream | Tolerance can consume prediction quality even with valid cells |
| Drift and recurrence | A previously valid fold becomes invalid |
| Rare witness | Scarce contexts need uncertainty, not forced pooling |
| Budget pressure | 96 contexts exceed a 32-context retention budget |

Exact H=1 candidate partitions fail H=3 audits in the delayed, action-dependent,
and stochastic-aliasing worlds. Nuisance and chain H=1 equality partitions pass.
That result illustrates why a short-horizon learner cannot claim full sufficiency.

## Learner and memory semantics

The learner imports no oracle/world code. It receives only an opaque observable
context, a declared probe and a binary observation. Ground-truth probabilities
remain in evaluation. Probe domains are fixed and checked. Unseen contexts give
a neutral 0.5 forecast; this is a probability estimate, not an identity assertion.

A cell pools predictions, but each member retains its own bounded observation
window and success count. Rebuilds can split a cell when evidence changes. Thus
folding reduces active prediction cells, **not retained statistical storage**.
Eviction discards evidence; it is counted explicitly. An evicted context must
relearn from fresh data, and has no promised lossless historical recovery.

Memory is O(C P W) binary samples plus bounded indexing, independent of stream
length for fixed context/probe/window caps. Folding search is O(C² P) per rebuild;
count updates are O(1), while prediction/eviction may scan at most C members.
These are object-count bounds; arbitrary context string byte lengths are not
bounded by the API. There is no dependency-cone theorem or lossless infinite
memory claim.

Complete-link folding prevents transitive chains in *empirical* profiles.
Statistical gates use conservative Hoeffding radii with a summable rebuild error
allocation. Their interpretation requires independent, stationary Bernoulli
samples; the real time series and changing regimes do not satisfy that contract.
Certificates can become stale between rebuilds or at a regime change. Reported
unsafe-fold rates use the evaluator's current probabilities, not the estimates.
No forecast is advertised as a perpetual safety certificate.

Ablations change one property relative to `empirical_fold`: enable confidence,
allow similarity chaining, prohibit reopening, or extend the window to 10,000.
`no_fold` is the simple bounded context-count baseline. Its confidence flag is
irrelevant because it never attempts a merge. All policies are serialized in the
results. Development and held-out streams share the generator family but have
disjoint seeds; there was no hyperparameter search. Promotion uses these pilot
results, so later confirmatory evaluation needs fresh seeds and worlds.

## Cortex usage

```python
from cortex import EvidenceAtom, PredictiveEvidenceAdapter

adapter = PredictiveEvidenceAdapter(("down", "up"))  # folding off by default
context = ("0", "1")
p_up = adapter.predict(context)  # score before observing outcome
adapter.observe(EvidenceAtom(context, "up", event_time=12, arrival_time=12))
```

The adapter accepts a declared two-value relation channel. It does not assign
meaning to arbitrary relations or consume hidden labels. Late observations do
not retrain the current predictive profile. Event-time frontiers are evicted with
their contexts. Equal-time events count separately; this adapter does not replace
the closed loop's provenance/dependence inference. Confidence weights and source
reliability are not modeled by this binary channel.

## Reproduction and evidence

From the repository root, Python 3.11+; no third-party dependency is needed for
the new tests or campaign:

```bash
PYTHONPATH=python:. python -m unittest discover -s tests -p test_sl_oracle.py -v
PYTHONPATH=python:. python -m research.sl_bench.run --output benchmarks/results/sl_bench_v0_1.json
```

The 15 tests cover independent state-path summation against exact trace
probabilities, Bell numbers/unique enumeration, exhaustive stable-memory
minimality for all 216 deterministic three-state binary-output sources,
relabeling, horizon refinement, semantic-role pseudometric cases, witnessed
rejection, confidence/domain checks, reopening, bounded storage and adapter
behavior. Another 16 existing closed-loop/regime checks passed by direct
source-tree invocation. The native Rust extension was unavailable locally;
no native/full-suite test result is claimed. CI adds the focused oracle suite.

The result JSON retains all 360 synthetic runs (2.16 million paired event
updates), policies, exact frontiers, per-world summaries and real forecasts'
aggregate metrics. The companion v1 computation manifest records source hashes,
data hash, base commit, command, environment and result checksum. It passed the
computation-audit validator. Its legacy schema lacks enforced process resource
limits; algorithmic caps are explicit. Hardware timings are descriptive, not a
controlled performance benchmark. Inputs are hashed after the run and were not
modified while it ran; this is weaker than a bounded runner's before/after hashes.

## Ethical and epistemic boundary

The real data is public-domain observational solar data and no action is applied
to people or external systems. The theory's non-identification principle is
reflected in scoped certificates, visible counterexamples and retention limits.
A benchmark pass is evidence about an implementation on these finite tests,
not proof that its articulated model exhausts the source.

## Remaining research gates

Before a production compression promotion: require independent world families,
explicit forecast-loss budgets as well as profile diameter, cheaper adaptive
uncertainty estimates, multi-step learned probes, and measured net storage savings
including the evidence needed to reopen. Add stochastic controlled sources to
learner evaluation, higher-order binding and learned invariance modules, then a
larger real stream with credible task-specific baselines. Native integration
requires a separate allocation/lifecycle design. None of these is marked done by
this pilot.
