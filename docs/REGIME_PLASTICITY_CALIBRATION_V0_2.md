# Regime Plasticity Calibration v0.2

Status: **calibration campaign**.

Relational Regime Plasticity v0.1 already removes the lifetime-evidence
dependence measured by Plasticity Frontier v1. This follow-up asks a stricter
question:

> Can the regime parameters be selected from an explicit stability/plasticity
> objective rather than hand tuned?

## Meaning of "optimal"

There is no universal scalar optimum for plasticity. A system can always adapt
faster by becoming easier to destabilize.

v0.2 therefore defines optimality **relative to a declared safety contract**.

Candidate configurations must first satisfy hard constraints:

- no false switch under stationary contradiction noise through 30%;
- no regime replacement from contradiction bursts of length 1, 2, 4, or 8;
- bounded persistent-change detection;
- bounded recurrent A -> B -> A switching;
- preserved regime archive/history.

Only configurations satisfying those constraints are ranked for adaptation
speed.

## Search space

The calibration sweeps 432 low-capacity configurations over:

- recent decay;
- challenge threshold;
- challenge patience;
- acceptance threshold;
- acceptance patience;
- minimum candidate observations.

Challenge hysteresis remains fixed at 0.05 for this campaign.

The search is deliberately small and interpretable; it is not a black-box
optimizer.

## Training objective

Among safe candidates, lower score is better.

The score combines:

- persistent regime-switch lag;
- sensitivity of lag to historical mass;
- time to leave the old exact articulation;
- recurrent A -> B -> A lag;
- recovery after transient contradiction bursts;
- time spent outside STABLE under stationary 20% and 30% noise.

False switches are not traded against speed: they are a hard rejection.

## Held-out validation

The selected configuration is frozen before evaluation on separate seeds.

Validation expands the stress envelope to:

- prior evidence masses from 4 through 512;
- stationary contradiction noise from 5% through 35%;
- 40% contradiction from a low-reliability source;
- contradiction bursts through length 24;
- recurrent A -> B -> A shifts.

The selected point is also compared with the hand-tuned v0.1 defaults.

## Promotion rule

A calibrated configuration is promoted only if it:

1. passes every held-out safety gate;
2. does not worsen stationary-noise false-switch safety;
3. matches or improves mean regime-switch lag;
4. belongs to a non-trivial neighborhood of safe parameter settings.

The final requirement guards against a brittle razor-thin optimum.
