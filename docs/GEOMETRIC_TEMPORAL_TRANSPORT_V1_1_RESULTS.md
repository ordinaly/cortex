# Geometric Temporal Transport v1.1 — Results

Status: **PASS**

The benchmark tests whether temporal identity can survive higher-order rearticulation when raw full-joint representation similarity is deliberately misleading.

## Holdout seeds 24–47 — full probe set

Across 24 independent seeds (48 source identity decisions):

- geometric identity accuracy: **100%**;
- naive full-joint nearest-neighbor accuracy: **0%**;
- mean correct transport distance: **0.03329**;
- mean cross-identity distance: **0.16174**;
- mean correct-vs-cross distance margin: **0.12845**;
- minimum per-seed mean distance margin: **0.10020**;
- mean correct successor-kernel probability: **0.73549**;
- mean correct full-joint TV: **0.75045**;
- mean cross-identity full-joint TV: **0.48621**;
- mean probe overlap: **1.0**.

The full-joint metric is therefore not merely noisy: on this adversarial rearticulation it systematically prefers the wrong identity, while transport over protected singleton/pairwise geometry remains correct.

## Holdout seeds 24–47 — one non-marker probe removed

Both successor candidates lose one non-marker probe. Shared probe overlap is exactly `3/4`.

- geometric identity accuracy: **100%**;
- mean correct transport distance: **0.08415**;
- mean cross-identity distance: **0.25521**;
- mean distance margin: **0.17106**;
- minimum per-seed mean margin: **0.13711**;
- mean correct successor probability: **0.79556**;
- mean overlap: **0.75**.

In this reduced representation the naive full-joint baseline happens to recover identity because the adversarial higher-order coordinate has been projected away. The important result is that the geometric transport remains correct despite probe loss and applies the overlap penalty without breaking continuity.

## Exact construction sanity check

At population level:

- source A (marker 0.30, EVEN) to true A successor (marker 0.30, UNIFORM): full-joint TV = **0.50**;
- source A to wrong B successor (marker 0.70, EVEN): full-joint TV = **0.40**;
- source B (marker 0.70, ODD) to true B successor (marker 0.70, EVEN): full-joint TV = **1.00**;
- source B to wrong A successor (marker 0.30, UNIFORM): full-joint TV = **0.55**.

Thus raw representation similarity swaps both identities by construction.

Meanwhile all parity singleton and pairwise marginals are invariant under EVEN/ODD/UNIFORM rearticulation; the persistent marker is preserved. The v1.1 transport scale therefore correctly treats the higher-order change as **rearticulation**, not identity destruction.

## Interpretation

This is the first Cortex result in which `K` transports an articulation rather than a symbolic state ID. It supports the proposed interpretation:

`identity = stable transport of protected articulation geometry through change`.

The result does not yet solve semantic probe correspondence, object splits/merges, or learned transport maps under transformations. Those are the next extensions.
