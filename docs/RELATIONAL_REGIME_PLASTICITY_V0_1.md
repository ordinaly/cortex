# Relational Regime Plasticity v0.1

Status: **Pareto calibration passed; research default frozen**.

This milestone addresses the failure exposed by
[Plasticity Frontier v1](PLASTICITY_FRONTIER_V1.md): cumulative relational
evidence detects contradiction early, but a replacement articulation must
numerically overwhelm the full historical record before it can execute.

The prototype deliberately separates two timescales:

- **historical evidence** is durable, exact, and never decayed;
- **current-regime evidence** is exponentially recency-weighted and is used to
  assess a possible replacement articulation with the frozen v0.2 controller
  surfaces.

A separate bounded contradiction-pressure signal controls when the currently
accepted articulation is suspended. This gives an explicit hysteretic path:

    stable articulation
        -> change suspected
        -> unresolved
        -> replacement articulation

The mechanism is research-only. It does not modify `spec-v0.9.5`.

## Calibration protocol

The development campaign swept **270 policies** over:

- fast-evidence decay: 0.50 / 0.60 / 0.65 / 0.70 / 0.75 / 0.80;
- contradiction-pressure decay: 0.85 / 0.90 / 0.95;
- suspicion ratio: 0.04 / 0.05 / 0.06;
- challenger confirmation: 1 through 5 observations;
- maturity cap: 64 historical observations.

Every candidate was tested on:

- abrupt OLD -> NEW shifts after 4 / 8 / 16 / 32 / 64 prior OLD examples;
- contradictory bursts of 1 / 2 / 4 / 8 observations;
- OLD -> NEW -> OLD reversal;
- stationary 10% and 20% contradictory noise;
- persistent noisy regime shifts with 90% and 80% NEW observations;
- disjoint-relation locality;
- exact historical retention;
- fixed-vocabulary state growth over 10,000 additional observations.

A candidate was feasible only if it preserved the frozen v0.2 change-detection
latency, produced no declared false regime switches, survived every noisy-shift
case, remained reversible and local, retained exact history, and kept
fixed-vocabulary regime state bounded.

## Pareto result

Of the **270** policies, **105** were feasible and only **2** were
Pareto-optimal under the declared objectives.

| fast decay | pressure decay | suspicion ratio | confirmation | clean switch | p95 shift @ 90% NEW | p95 shift @ 80% NEW | mean recovery | stationary unresolved |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.90 | 0.05 | 1 | 9 | 20 | 34 | 6.25 | 0.1642 |
| **0.50** | **0.90** | **0.05** | **5** | **9** | **22** | **35** | **3.75** | **0.0921** |

The frozen research default is the second point. The knee selector minimizes
normalized distance to the ideal point over clean-switch lag, noisy-shift p95,
recovery cost, stationary unresolved rate, and reverse-switch lag.

The selected point gives up only 1 observation of p95 latency in the 80%-NEW
case and 2 observations in the 90%-NEW case relative to the other Pareto point,
while reducing mean recovery support from 6.25 to 3.75 and stationary
unresolved occupancy from 0.1642 to 0.0921.

## Selected policy

    fast_decay = 0.50
    pressure_decay = 0.90
    suspicion_ratio = 0.05
    confirmation_steps = 5
    maturity_cap = 64

The suspicion threshold is proportional to accepted historical maturity only up
to the cap. Historical evidence therefore remains fully preserved for audit and
recurrence, while lifetime age cannot create unbounded plasticity debt.

## Measured behavior

### Genuine regime shift

The cumulative baseline required:

| prior OLD evidence | baseline NEW -> EXECUTE | calibrated regime switch |
|---:|---:|---:|
| 4 | 72 | 9 |
| 8 | 143 | 9 |
| 16 | 285 | 9 |
| 32 | 570 | 9 |
| 64 | 1139 | 9 |

The old articulation still leaves `EXECUTE` after **1 / 1 / 1 / 2 / 4**
contradictory observations, exactly matching the Plasticity Frontier v1
detection contract. The replacement now reaches `EXECUTE` after **9**
observations for every tested prior mass.

This removes the measured lifetime-history proportionality in the declared
range without weakening early contradiction detection.

### Transient contradiction

For a mature 32-example relation:

| burst | deepest state | false switch | support to recover |
|---:|---|---|---:|
| 1 | EXECUTE | no | 1 |
| 2 | UNRESOLVED | no | 4 |
| 4 | UNRESOLVED | no | 5 |
| 8 | UNRESOLVED | no | 5 |

No burst caused a regime switch.

### Stationary noise

Across 8 seeds x 300 observations at each noise level:

- false switched steps: **0**;
- mean unresolved occupancy across the 10% and 20% conditions: **0.0921**.

The unresolved state is therefore doing real work: under stronger stationary
noise Cortex may suspend exact articulation without rewriting the accepted
regime.

### Noisy genuine shifts

Across the development seeds:

- 90% NEW: median switch latency **16.5**, p95 **22**, maximum **22**;
- 80% NEW: median switch latency **21**, p95 **35**, maximum **35**;
- failed switches within 300 observations: **0**.

The persistent-noise test is important because very short fast memory combined
with long confirmation can look excellent on a perfectly clean switch while
failing to identify a genuinely changed but imperfect world.

### Reversal and history

OLD -> NEW and NEW -> OLD both switched in **9 observations** on the reversal
fixture.

Historical counts were retained exactly. After 64 initial observations and
10,000 later observations, the audit history contained all **10,064**
observations while the two-relation fixture still used only **4 relation-indexed
regime cells**. This is bounded with respect to stream length for a fixed
relation vocabulary; it is not a claim that memory is independent of vocabulary
growth.

A disjoint control relation was unchanged throughout the shift.

## Baseline interpretation

Pure exponential decay adapts faster at the most aggressive settings, but decay
0.50-0.65 produced a false switch under the declared transient burst test.
The naive version also discards historical mass.

Keeping history beside the decayed signal fixes auditability but does not by
itself create an explicit change-suspicion boundary. The selected mechanism
therefore keeps all three roles distinct:

1. durable historical memory;
2. fast current-regime articulation;
3. hysteretic change detection and confirmation.

## Claim boundary

The result establishes a Pareto-superior plasticity mechanism **within the
declared relational-composition fixtures and parameter grid**. It does not
establish globally optimal plasticity, optimal change-point detection, or
robustness to arbitrary non-stationarity.

The next milestone should integrate this mechanism into
**Autonomy Robustness & Temporal Plasticity v1**, where source reliability,
missing/corrupted observations, broader relation structures, memory pressure,
and downstream reasoning consequences are evaluated together.
