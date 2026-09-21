# Articulation Controller Calibration v0.2

Status: **validated experimental learned metacognitive policy**.

## Goal

Articulation Controller v0.1 uses transparent fixed thresholds. v0.2 learns
small decision surfaces from consequences while preserving an inspectable
epistemic boundary.

The learned controller does **not** learn relation semantics. It learns only:

- when current Cortex evidence is ready for exact execution;
- whether rejected execution should remain open to exploration or be marked
  unresolved for a specific reason;
- whether native Cortex traversal or the LPU is the cheaper exact backend.

## Risk contract

Calibration cannot learn away the choice of acceptable execution risk. v0.2
therefore declares its oracle contract explicitly:

- future dominant-relation reliability floor: **0.75**;
- future source reliability floor: **0.80**;
- future structural-support floor: **0.55**;
- current Wilson bound uses **z = 1.0**;
- current resolved-evidence floor: **0.60**;
- execution threshold is selected for at least **0.93 validation precision**.

Exact execution remains conditional on the premises. `EXECUTE` does not mean
that a premise has become certain.

## Stratified future-evidence calibration

The synthetic calibration population is stratified across five regimes:

1. mature stable structure;
2. stable structure with sparse current evidence;
3. persistent relational conflict;
4. transient source ambiguity;
5. persistent weak structural support.

A large held-out future sample exposes the latent regime to the oracle, while
the learned controller receives only current Cortex-like evidence features.
This prevents calibration from simply copying v0.1's thresholds.

## Learned surfaces

v0.2 uses four deterministic monotone logistic surfaces.

### Execution readiness

Features:

```text
evidence maturity
rule confidence
positive rule margin
mean structural support
source support
source certainty
```

The threshold is selected on validation data under the explicit precision
contract. On the held-out test split:

- precision: **0.9405**;
- recall: **0.4422**;
- dangerous false-execute rate over all states: **0.0125**.

The controller is intentionally conservative: false execution is more costly
than continued exploration.

### Relational conflict

This surface asks whether evidence supports a persistent incompatibility between
competing relation hypotheses.

Features:

```text
evidence maturity
competitor confidence
conflict strength
low target confidence
source certainty
```

Held-out result:

- accuracy: **0.9533**;
- precision: **0.8508**;
- recall: **0.9747**.

### Structural weakness

This is intentionally separate from relational contradiction.

Features:

```text
evidence maturity
low structural support
source certainty
```

Held-out result:

- accuracy: **0.9864**;
- precision: **0.9698**;
- recall: **0.9912**.

The separation is important. A single unresolved surface previously conflated
moderate disagreement with persistent weak structure.

### Backend selection

Features:

```text
path scale
path-compression ratio
```

Training labels come from measured native Cortex-enumerator versus layered-LPU
execution cost on opaque relation lattices. The validation threshold minimizes
runtime regret.

In the successful CI run the held-out backend set achieved:

- accuracy: **1.000**;
- mean runtime regret: **1.000x**;
- median runtime regret: **1.000x**;
- maximum runtime regret: **1.000x**.

These timing-derived results are environment-specific; the learned structural
surface, not the absolute timings, is the artifact of interest.

## Three-state holdout

On 1,200 held-out future-evidence episodes:

- three-state accuracy: **0.7075**;
- dangerous false execution: **0.0125**.

The confusion is intentionally asymmetric: many safe-but-not-yet-admitted
episodes remain `EXPLORE` rather than being executed prematurely.

## Native Cortex holdout

The learned bundle was then tested on new native Cortex worlds, separate from
the synthetic calibration episodes.

Across **40 clean cases**:

- execution rate: **1.000**;
- answer accuracy: **1.000**;
- memory isolation: **1.000**;
- binding conflicts: **0**;
- selected knowledge resolution: **epsilon = 0.65** in all cases.

The learned backend selected Cortex for 10 cases and LPU for 30 cases in this
run. Unlike v0.1's fixed heuristic, this boundary is learned from measured
execution cost.

Epistemic controls:

```text
clean evidence      -> EXECUTE
20% contradiction   -> EXPLORE
40% contradiction   -> UNRESOLVED
ambiguous source    -> EXPLORE
```

At 20% contradiction the learned conflict probability was about **0.217**;
at 40% it rose to about **0.708**. Thus moderate disagreement remains open
rather than being prematurely frozen, while strong conflict is explicitly
recognized.

## Architectural invariants

Calibration does not grant the controller semantic authority:

- relation names are absent from all learned features;
- LPU output cannot increase premise confidence;
- no exact execution occurs without execution admission;
- conflict and structural weakness are distinguished rather than collapsed;
- multiple legitimate targets remain valid;
- execution remains read-only with respect to Cortex memory;
- exact-witness requests can choose LPU only after epistemic admission;
- evidence-resolution filtering remains upstream of exact execution.

## Interpretation

v0.2 supports a learned metacognitive boundary:

```text
Cortex evidence
      |
      v
learned execution surface
  | admitted
  +------------------> exact execution
  |
  | rejected
  v
relational-conflict surface ----+
                                 +--> UNRESOLVED
structural-weakness surface ----+
  |
  +-- neither --> EXPLORE
```

The result is deliberately low-capacity and auditable. More expressive
controllers should be considered only if future domains demonstrate a failure
that cannot be represented by cause-specific monotone surfaces.
