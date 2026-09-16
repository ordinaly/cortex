# Cortex v0.9.5-alpha — Compute-Optimized Refinement Runtime

## Scope
Cortex v0.9.5-alpha is a compute-only refinement of v0.9.4-alpha. It does not add a new cognitive operation. The goal is to preserve the v0.9.4 update/split/merge/novelty behavior while reducing avoidable runtime work.

## Optimizations

### 1. Cached centroid carrier
The prototype centroid matrix is retained across observations instead of rebuilt with `stack` on every distance query. Under ordinary learning only the current prototype centroid can drift, so only one cached row is synchronized. Structural operations (append, recompress, split, merge) invalidate the cache.

This removes repeated allocation and reconstruction while leaving the distance rule unchanged.

### 2. Strided residual-curvature surveillance
v0.9.4 updates the signed residual-curvature matrix

Q_i <- (1-r) Q_i + r e_t z_t z_t^T

on every tensor-awake observation. v0.9.5-alpha samples every second tensor-awake observation. The effective EMA coefficient is corrected as

r_eff = 1 - (1-r)^2,

so the monitor retains approximately the same decay horizon while halving dense outer-product work.

This is an approximation only in the *surveillance statistic*. Split promotion still requires the unchanged prospective shadow-validation contract.

### 3. Event-directed refinement bookkeeping
Split-launch logic is evaluated only when the curvature monitor actually advances. Shadow validation runs only for active candidates. Merge monitoring runs only when a split lineage exists. Per-step stale-candidate scanning is removed because structural deletion paths already purge invalid candidate references.

## Safety contracts
v0.9.5-alpha preserves the structural meaning of v0.9.4-alpha:
- parametric-update fixture: no split,
- heterogeneous-concept fixture: one split,
- genuinely novel fixture: new regime rather than split,
- split-then-collapse fixture: later merge,
- real streams: identical structural state and predictions to v0.9.4 in the tested six streams.

## Complexity
Let B be the number of prototypes, d the feature dimension, and rho_T the v0.9.3 tensor-awake fraction.

Distance work remains O(Bd), but carrier reconstruction is amortized across steps.

Dense curvature work remains O(d^2) when sampled, but its average frequency is reduced by approximately 1/2 at the frozen stride. Thus the surveillance contribution is roughly

O((rho_T / 2) d^2)

rather than O(rho_T d^2).

The next scaling frontier is to replace the dense Q_i itself with a certified low-rank/streaming sketch, reducing the d^2 memory/update term rather than merely its duty cycle.
