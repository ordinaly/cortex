# Autonomy Robustness & Temporal Plasticity v1

Status: **Phase A baseline campaign active / preregistered**.

This program integrates the frozen **Relational Regime Plasticity v0.1**
mechanism into a broader autonomous-stream setting before any further
plasticity tuning.

The guiding question is:

> Does Cortex remain safely plastic when evidence is incomplete, corrupted,
> temporally changing, source-dependent, structurally heterogeneous, and
> resource-bounded?

The program is research-only and does not modify `spec-v0.9.5`.

## Experimental discipline

Phase A makes **no changes** to the frozen Pareto-selected regime policy:

```text
fast_decay         = 0.50
pressure_decay     = 0.90
suspicion_ratio    = 0.05
confirmation_steps = 5
maturity_cap       = 64
```

The first goal is characterization, not optimization. New mechanisms are only
introduced after the baseline exposes a specific failure with a measurable
consequence.

This avoids tuning source weighting, temporal decay, or memory policy against
the same fixtures used to claim improvement.

## Phase A architecture

The first integration layer is an opaque relation-signature bank:

```text
chronological observations
        |
        +--> context A --> RelationalRegimeTracker
        |
        +--> context B --> RelationalRegimeTracker
        |
        +--> context C --> RelationalRegimeTracker
        |
        +--> ...
```

A context is an opaque tuple describing a learned relation path/signature. It is
not a semantic predicate.

Each local tracker retains the already validated separation between:

1. exact durable historical evidence;
2. fast current-regime evidence;
3. bounded hysteretic contradiction pressure.

The bank adds only experiment-level chronology and locality.

## Missing observations

A missing observation is an explicit event with no relation payload.

Phase A uses **event-time evidence semantics**:

- missing evidence does not confirm a regime;
- missing evidence does not contradict a regime;
- missing evidence does not decay the fast channel;
- missing evidence does not increment historical evidence.

The benchmark therefore reports both:

- **observed-evidence latency**;
- **wall-clock latency**.

This distinction is deliberate. If a later experiment demonstrates that real
elapsed time itself should weaken evidence, that must be introduced as a new
mechanism and tested separately.

## Source reliability

Every observation can carry:

- source identity;
- declared source reliability.

Phase A records this metadata but the frozen policy **does not consume it**.

A preregistered diagnostic creates a case where:

- a high-reliability minority supports the new regime;
- a low-reliability raw-count majority supports the old regime;
- reliability-weighted support favors the new regime.

This test is not part of the Phase A pass/fail gate because failure is expected
to expose whether source-aware evidence is genuinely necessary. It must not be
silently fixed before the baseline is measured.

## Perturbation axes

Phase A covers:

1. clean abrupt regime change;
2. 30% and 60% missing observations;
3. 10% and 20% corrupted observations during a genuine shift;
4. stationary 10% and 20% contradictory noise;
5. gradual OLD -> NEW drift;
6. source-reliability conflict;
7. simultaneous independent contexts with shift / stability / reversal;
8. fixed-context, fixed-relation-vocabulary streams under long chronology.

Later phases will add native hypergraph reasoning, entity pressure, broader
out-of-distribution relation structures, and exact LPU consequences.

## Downstream safety metric

Switch latency alone is not sufficient.

For every step with a known test-world regime, the harness records:

- **safe exact execution**: `EXECUTE` and accepted relation matches the world;
- **unsafe exact execution**: `EXECUTE` while a stale/wrong relation is
  accepted;
- **abstention**: `EXPLORE` or `UNRESOLVED`.

This lets us distinguish a fast but reckless learner from a slower learner that
uses uncertainty correctly.

## Frozen Phase A gates

The initial baseline must preserve already validated properties:

- clean replacement remains bounded at <= 16 observed events;
- missingness does not change observed-evidence switch latency;
- 10% and 20% corrupted genuine shifts still switch within 300 observed events;
- stationary 10% and 20% contradiction produces no false regime switch;
- concurrent relation contexts remain local;
- fixed context/relation vocabulary does not grow regime state with stream
  length.

The source-reliability trap and gradual-drift results are diagnostic and are not
used to tune the frozen baseline.

## Phase B decision rule

A new mechanism is justified only if Phase A exposes a material weakness.

Likely candidates include source-weighted evidence, learned source reliability,
or explicit elapsed-time semantics, but none is assumed in advance.

Any Phase B candidate must:

- improve the diagnosed robustness failure;
- preserve or improve unsafe-execution rate;
- preserve the v0.2 early contradiction-detection contract;
- keep clean/noisy regime-switch latency within the declared non-regression
  envelope;
- preserve exact historical auditability;
- preserve legitimate multi-valued relations;
- preserve context locality;
- keep state bounded in stream length for a fixed vocabulary.

## Claim boundary

Passing Phase A will establish only that the frozen local plasticity mechanism
survives the declared robustness perturbations when composed across opaque
relation contexts.

It will not establish source-aware reasoning, arbitrary temporal semantics,
native memory-pressure robustness, natural-language autonomy, or unrestricted
world modeling.

See [RELATIONAL_REGIME_PLASTICITY_V0_1.md](RELATIONAL_REGIME_PLASTICITY_V0_1.md)
and [CORTEX_AUTONOMY_CAMPAIGN_V1.md](CORTEX_AUTONOMY_CAMPAIGN_V1.md).


## Phase A development result

The first 12-seed development run passed every frozen Phase A non-regression
gate.

### Clean abrupt change

The frozen policy switched in exactly **9 observed events** on every seed.

Because the prior-32 v0.2 contradiction detector intentionally does not suspend
execution until the second contradictory event, the clean stream spent one of
the nine pre-switch events executing the stale incumbent. Mean unsafe-execution
rate over the pre-switch window was therefore **0.1111**.

### Missing observations

Missingness did not alter evidence-count latency:

| missing rate | mean observed switch | mean wall switch | p95 wall switch | mean unsafe-execute rate |
|---:|---:|---:|---:|---:|
| 30% | 9 | 13.33 | 16 | 0.1626 |
| 60% | 9 | 23.83 | 30 | 0.1860 |

This validates the declared event-time semantics but exposes an important
downstream limitation: a long observation gap leaves the previous
`EXECUTE` permission intact. Memory is not being corrupted, but exact
execution can remain stale for longer in wall time.

The next phase should therefore distinguish **memory persistence** from
**execution freshness**. Missing data need not erase or decay historical
evidence, while a separate freshness gate may still suspend exact execution
after sufficient unobserved time.

### Corrupted genuine shifts

Both declared corruption levels switched on every seed:

| corruption | mean observed switch | p95 observed switch | failures | mean unsafe-execute rate |
|---:|---:|---:|---:|---:|
| 10% | 15.75 | 26 | 0 | 0.0924 |
| 20% | 27.33 | 58 | 0 | 0.0758 |

The increasing switch latency is expected under conflicting observations. The
controller spends more of the interval abstaining rather than executing a stale
regime.

### Stationary contradictory noise

Across 300 steps per seed:

- 10% noise: **0 false switches**, mean abstention **0.0142**, unsafe exact
  execution **0**;
- 20% noise: **0 false switches**, mean abstention **0.1286**, unsafe exact
  execution **0**.

This preserves the stability side of the previously calibrated frontier.

### Gradual change

For a 40-step OLD -> NEW probability ramp:

- mean switch wall step: **38.67**;
- p95 switch wall step: **43**;
- mean abstention: **0.1453**;
- mean unsafe exact execution: **0.00573**.

The frozen mechanism therefore handles this gradual fixture conservatively:
uncertainty rises before the final re-articulation and stale exact execution is
rare.

### Source-reliability diagnostic

The preregistered source-quality trap produced the clearest new failure.

Each cycle contains a high-reliability minority supporting NEW and a
low-reliability raw-count majority supporting OLD. Cumulative
reliability-weighted support favors NEW.

The frozen Phase A policy does not consume reliability metadata:

- reliability-weighted oracle: **NEW**;
- final accepted NEW: **0 / 12**;
- final accepted OLD: **12 / 12**.

This establishes a concrete source-reliability blind spot. Phase B is therefore
justified in testing source-aware evidence rather than adding it by assumption.

### Locality and bounded state

The simultaneous shift / stable / reversal context fixture preserved locality
on every seed.

For the fixed-vocabulary memory test, **48 contexts** occupied **192 regime
cells** before a further 6,000 observations and **192 cells** afterward.
State therefore remained bounded in stream length for the fixed declared
context/relation vocabulary.

## Phase B target

Phase B should attack two measured limitations while leaving the successful
parts of Phase A untouched:

1. **source-aware evidence fusion** — reliable observations should influence
   current-regime articulation more than weak observations without rewriting
   raw historical audit evidence;
2. **execution freshness** — missing evidence may age permission to execute
   without deleting or contradicting the accepted historical regime.

Candidate mechanisms should be compared rather than assumed. At minimum, test
linear reliability weighting against an unweighted control and calibrate a
freshness/abstention boundary separately from regime-memory decay.

Promotion requires improvement on the source-quality and missingness safety
diagnostics while preserving the clean 9-event switch, stationary-noise
stability, corrupted-shift success, locality, reversal, and fixed-vocabulary
bounded-state properties.
