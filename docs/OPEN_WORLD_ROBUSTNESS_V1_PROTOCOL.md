# Cortex Open-World Robustness v1

Status: **Phase 1 preregistration / benchmark construction**.

Protocol ID: \`open-world-robustness-v1\`.

## Purpose

This campaign is the first deliberately integrated falsification campaign after
Cortex Autonomy v1 and Autonomy Robustness & Temporal Plasticity v1.

The question is no longer whether Cortex can perform one isolated structural
operation. The question is whether a frozen Cortex can maintain a useful,
revisable world model when several forms of uncertainty and novelty interact.

The benchmark remains structured. Raw language and raw perception are excluded
so failures can be attributed to the cognitive architecture rather than an
encoder.

## Scientific rule

The benchmark is not designed to be passed by the current system.

A failure is a valid result.

Cortex mechanisms may be changed during development only on development worlds.
After the candidate, metrics, and gates are frozen, the official worlds are
derived from the merge commit SHA and the official outcome is immutable.
Any post-official correction requires a new protocol version and a new seed
derivation.

## Campaign structure

The umbrella campaign has five stages.

### OWR-A — Epistemic Sources

Question:

> Can Cortex infer which sources deserve current evidential weight when source
> reliability is hidden?

The world contains:

- multiple opaque sources;
- contextual source reliability;
- a correlated/copied weak-source cluster;
- strong independent sources;
- source degradation after a stable history;
- target contexts where a copied numerical majority can be wrong.

The learner receives source identity and claims only. It does **not** receive
the hidden reliability values or copying graph.

Primary future metrics:

- current-truth accuracy / selective risk;
- learned reliability calibration;
- copied-evidence overcounting;
- source-drift detection latency;
- trust-revision latency;
- false certainty.

The current Autonomy Phase-B policy is an important control because its source
weights currently require externally supplied reliability. Running all sources
at unit reliability should expose that limitation rather than hiding it.


Development fixture hardening before any official run:

- source reporting order is randomized independently per context/round, so
  recency cannot privilege a fixed source position;
- source reliability and source-dependence metadata remain evaluator-only;
- source drift occurs only after a genuine stable calibration period;
- source-trust evaluation transfers from **eight calibration contexts** to
  **six fresh target contexts** that have never appeared during calibration;
- four fresh targets contain a coherent copied-majority trap: one weak root and
  two strongly correlated copies oppose two independent strong sources;
- two fresh targets are controls where the source ensemble remains compatible
  with the truth.

#### Rejected development fixture v0

The first development fixture kept the same target contexts throughout a long
stable history and introduced copied misinformation only late.

Across 12 development worlds, the current source-blind Cortex remained
conservative: final control coverage was about **0.708**, while copied-majority
trap coverage fell to about **0.104**, with no pooled resolved errors. However,
a cumulative context-local majority still reached **1.000 final trap accuracy**
because old correct target history overwhelmed the later bad reports.

That fixture therefore tested abstention/plasticity more than transferable
source epistemics and is rejected before any official run.

#### Development fixture v1

The replacement removes that escape hatch. Cortex first observes source
behavior on calibration contexts. Only then are fresh target contexts
introduced. A target-local historical majority therefore has no earlier truth
to reuse.

The benchmark also includes a public-data dependence baseline that may infer
strong source correlation from calibration agreement but receives no hidden
truth labels, reliability values, or copy graph. This verifies that the
dependence signal is present in learner-visible data rather than only in the
oracle.

Performance gates remain unfrozen until this revised development
characterization is complete.

### OWR-B — Temporal Disorder

Question:

> Can Cortex maintain the current articulation when event time and observation
> time differ?

The world contains:

- missing observations;
- delayed observations;
- out-of-order arrival;
- duplicated observations;
- abrupt regime changes;
- recurrence \`A -> B -> C -> A\`.

The benchmark keeps both event time and arrival time. The learner sees arrival
order; the oracle knows the latent event-time regime.

Primary future metrics:

- current-regime accuracy / coverage;
- false certainty under stale evidence;
- switch latency measured in event time and observation time;
- duplicate sensitivity;
- recurrence/reactivation latency;
- \`reactivation latency / first-learning latency\`.

A strong memory result would satisfy, on average,

\`\`\`text
reactivation latency < first-learning latency
\`\`\`

without increasing false switches.


Development hardening before any official run:

- queries are placed at regime boundaries and at +3, +10, and later stable
  checkpoints, so generous recovery time cannot hide transition failures;
- event time and arrival time remain distinct learner-visible fields;
- duplicate provenance remains learner-visible;
- the current Cortex temporal adapter is characterized both unchanged and with
  a public-provenance deduplication ablation;
- a public event-time recent-evidence baseline tests whether event timestamps
  contain useful information the current arrival-order adapter leaves unused;
- transition latency is reported separately for the two novel regime switches
  and the later recurrence to the original relation.

### OWR-C — Compound Novelty

Question:

> Can Cortex represent a structural explanation that requires more than one
> simultaneous change?

The world includes cases requiring combinations of:

- new entity type;
- new relation;
- new regime.

It also includes a deliberately ambiguous case where two compound explanations
cannot be separated from available evidence.

A single novelty label is not sufficient for this stage.

Primary future metrics:

- exact component-set precision / recall;
- unnecessary structural edits;
- edit locality;
- ambiguity preservation;
- post-edit stability;
- false persistent edit rate.

The correct output for the frozen ambiguous case is \`UNRESOLVED\`.

### OWR-D — Memory & Interference

Question:

> Under a long mostly irrelevant stream, does bounded memory preserve useful
> structure and degrade safely when it cannot?

Development worlds contain at least 3,000 events; official worlds use a longer
stream. Only about 3.5% of events belong to salient long-lived contexts.

The stream includes:

- hundreds of distractor contexts;
- sparse early salient structure;
- one temporary salient regime;
- later recurrence to the original regime;
- queries long after the supporting evidence.

Primary future metrics:

- retained salient-query accuracy / coverage;
- distractor sensitivity;
- memory growth;
- query cost;
- false certainty under memory pressure;
- recurrence recovery;
- graceful degradation toward \`UNRESOLVED\`.

Hard capacity exhaustion must not be converted silently into confident wrong
answers.

### OWR-E — Autonomous Hidden Physics World

Question:

> Can the integrated architecture discover and maintain useful structure in a
> world without being told which cognitive capability is being tested?

The hidden world uses opaque relation symbols whose latent composition obeys a
permuted cyclic law:

\`\`\`text
latent(a + b + regime_offset) mod 7
\`\`\`

Surface relation tokens carry no semantic information.

The world combines:

- hidden algebraic composition;
- held-out relation-pair contexts;
- source disagreement;
- correlated sources;
- source degradation;
- missingness;
- delayed observations;
- multiple regimes;
- recurrence to an earlier regime;
- an explicitly ambiguous query.

Queries are issued across regimes. Some query contexts are never directly
observed and therefore require structural generalization rather than direct
memory.

OWR-E is the composite examination. It must not be used as the tuning target
for mechanisms introduced to solve A-D.

## Phase 1 implementation boundary

Phase 1 implements only:

- deterministic world generation;
- public learner-facing observations;
- separate hidden oracle state;
- deterministic seed derivation;
- structural integrity checks;
- development/official scale separation.

It intentionally does **not** change Cortex.

It also intentionally does not freeze Cortex performance thresholds yet. Those
thresholds are frozen after development characterization and before any
official run.

## Seed sealing

Development worlds are derived from arbitrary development labels.

Official worlds must use the merge commit SHA of the frozen benchmark/candidate
as seed material:

\`\`\`text
seed = H(
  "open-world-robustness-v1"
  || stage
  || world_index
  || merge_commit_sha
)
\`\`\`

The implementation uses SHA-256 and the first 63 bits of the digest.

Official mode rejects seed material that is not a 40-character hexadecimal
commit SHA.

This is not cryptographic secrecy. Its purpose is procedural: the exact
official worlds do not exist until the frozen candidate and benchmark are
merged.

## Information boundary

Learner-facing observations contain:

- opaque source ID;
- opaque context symbols;
- observed relation or explicit missing value;
- event time;
- arrival time;
- duplicate provenance when the duplicate itself is observable.

Learner-facing queries contain only:

- query ID;
- query time;
- opaque context.

Harness tags, hidden truth, expected answers, answerability labels, source
reliability, copying graphs, latent laws, and regime schedules are evaluator
state only.

The hidden oracle may contain:

- true source reliability;
- copying/dependency graph;
- latent relation-token mapping;
- hidden regime schedule;
- true composition law;
- expected compound novelty components.

Adapters must not pass hidden oracle fields into Cortex.

Any adapter that consumes oracle-only data outside evaluation invalidates the
run.

## Metrics to freeze before official execution

The final preregistration must include at least:

- resolved accuracy;
- coverage;
- selective risk;
- Brier score where probabilistic confidence exists;
- calibration error;
- false-certainty rate;
- unresolved precision;
- structural edit precision / recall;
- edit locality;
- revision and recovery latency;
- recurrence/reactivation latency;
- unaffected-memory preservation;
- state/memory growth;
- runtime/query cost;
- planning regret and unnecessary-replan rate where planning is included.

No single aggregate "Cortex score" is authoritative.

Risk/coverage curves are a primary surface.

## Baselines

At minimum the development campaign should characterize:

- direct memory;
- recency/majority tracker;
- source-blind temporal Cortex;
- Cortex without regime plasticity;
- Cortex without abstention;
- simple probabilistic source/changepoint reference where appropriate;
- oracle references that know hidden source reliability or hidden regime.

An oracle is an achievable reference, not a competitor Cortex is expected to
beat.

LPU-on/LPU-off comparisons must preserve semantic conclusions whenever exact
execution is involved.

## Negative-result policy

1. Benchmark bugs may be fixed during development.
2. Mechanisms may be tuned only on development worlds.
3. Performance gates are frozen before official world generation.
4. OWR-E must not be used for iterative tuning after it becomes the official
   composite examination.
5. If an official gate fails, the failure remains in the research record.
6. A corrected mechanism requires a new protocol version and newly derived
   official worlds.

## Claim boundary

Even a full pass would support only:

> teacher-free open-world structural reasoning under declared synthetic
> structured streams with interacting uncertainty, novelty, temporal disorder,
> memory pressure, and hidden relational regularities.

It would **not** establish:

- raw language understanding;
- raw sensory perception;
- unrestricted scientific discovery;
- general intelligence;
- real-world adversarial robustness;
- open-ended ontology growth at internet scale.

Those require later representation-discovery and perceptual-bridge campaigns.

## Planned implementation

Phase 1:

- \`research/open_world_robustness_world.py\`
- \`research/open_world_robustness_campaign.py\`
- \`tests/test_open_world_robustness.py\`
- manual research workflow entry

Phase 2:

- Cortex adapters and declared baselines for OWR-A through OWR-D;
- development characterization;
- frozen metrics and promotion gates.

Phase 3:

- freeze candidate + protocol;
- merge;
- derive official seeds from merge SHA;
- run A-D official diagnostics;
- run OWR-E composite examination exactly once for that protocol version.
