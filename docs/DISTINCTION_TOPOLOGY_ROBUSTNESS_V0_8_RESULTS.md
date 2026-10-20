# Distinction Topology Robustness v0.8 — Results

## Verdict

**PASS** on the controlled noisy development/holdout benchmark.

The experiment supports a nontrivial temporal-hysteresis region for v0.7 pairwise distinction topology. The result is not monotone in hysteresis strength: insufficient hysteresis increases topology error, while excessive hysteresis causes stale-edge lock-in and severe plasticity failure.

## Latent topology

The evaluator-only ideal complementarity values are:

### Before the switch

| Edge | Ideal Gamma |
|---|---:|
| `(0,1)` | 1/6 |
| `(0,2)` | 1/3 |
| `(1,2)` | 1/6 |

At an activation threshold near 0.30, the intended graph contains only `(0,2)`.

### After the switch

| Edge | Ideal Gamma |
|---|---:|
| `(0,1)` | 1/6 |
| `(0,2)` | 1/6 |
| `(1,2)` | 1/3 |

The same three neurons therefore need to rewire from `(0,2)` to `(1,2)`.

## Calibration search

- 22 valid `(edge_on, edge_off)` threshold pairs were evaluated.
- Development seeds: `0..11`.
- Holdout seeds: `12..23`.
- Two development policies passed every development gate.
- The safety-first tie-break selected:

```text
edge_on  = 0.30
edge_off = 0.20
```

This is a reference result for the controlled noisy fixture, not a claim of universal optimality.

## Development result

For `(0.30, 0.20)` across 12 development seeds:

- stationary topology error: **8.4375%**;
- mean stationary edge flips: **2.0833**;
- rewiring success: **12/12 (100%)**;
- mean rewiring delay: **41.75 ticks**;
- late post-switch topology error: **12.5%**;
- mean total edge flips: **9.25**.

All development acceptance criteria passed.

## Holdout result

The selected policy was then frozen and evaluated on 12 unseen seeds:

- stationary topology error: **8.6458%**;
- mean stationary edge flips: **1.9167**;
- rewiring success: **12/12 (100%)**;
- mean rewiring delay: **52.33 ticks**;
- late post-switch topology error: **9.1667%**;
- mean total edge flips: **13.33**.

The mean rewiring delay remains below the 64-sample sliding-window horizon.

## Ablation: no hysteresis

Using the same activation threshold but setting `edge_off=edge_on=0.30` gives:

- stationary topology error: **9.5833%**;
- mean stationary edge flips: **2.0833**;
- rewiring success: **12/12 (100%)**;
- mean rewiring delay: **52.33 ticks**;
- late post-switch topology error: **25.625%**;
- mean total edge flips: **15.5**.

Thus the calibrated band preserves the same mean rewiring latency while substantially reducing late topology error and total churn.

## Ablation: excessive hysteresis

Keeping `edge_on=0.30` but lowering `edge_off` to `0.10` gives:

- stationary topology error: **36.5625%**;
- mean stationary edge flips: **0.5**;
- rewiring success: **2/12 (16.67%)**;
- mean delay among the two successful rewires: **109 ticks**;
- late post-switch topology error: **94.1667%**;
- mean total edge flips: **3.1667**.

This is the important negative control: low churn by itself is not desirable. The graph can become quiet because wrong edges are locked in.

## Interpretation

The result establishes an empirical stability/plasticity frontier for pairwise distinction topology:

```text
edge_off too high  -> topology chatter / noisy edge turnover
edge_off too low   -> stale-edge lock-in / lost plasticity
middle band        -> stable but still rewritable topology
```

The notable point is that the vertices never change. Only their learned complementarity changes. Cortex therefore demonstrates, on this controlled streaming problem, the intended architectural behavior:

```text
same distinction population + changed phenomenon -> changed relational topology
```

while retaining finite-window noise robustness.

## Decision

v0.8 is strong enough to merge as a robustness milestone. The calibrated `(0.30, 0.20)` pair should remain a **research reference**, not a universal production default, until tested across additional world families.

The next milestone may now investigate higher-order structure, but it should not create triangles merely from three pairwise edges. A 2-simplex should require measurable triple-level structure that is not reducible to the three pairwise relations.
