# Five-Operator Bench v0.1

Status: **oracle-assisted architecture benchmark / research-only**.

This benchmark is the first validation layer for the five-operator articulation loop introduced in `FIVE_OPERATOR_ARTICULATION_V0_1.md`.

It deliberately does **not** test autonomous learning yet. The point of v0.1 is narrower:

> If Cortex is supplied with faithful local distinction operators and the allowed potential simplices, does the shared `Delta/K/C/V/I` architecture preserve the structures that the theory says it should preserve?

Only after that question has a clean answer should the supplied operators be replaced by learned ones.

## Benchmark boundary

Evaluator-owned information may be used to construct oracle distinction neurons and expected answers. The online five-operator loop receives the resulting operators, observations, intervention records, orientation values and local views; it does not import hidden transition tables or oracle functions.

The benchmark reuses the finite Sigma-Lambda oracle under `research/sl_bench/` rather than creating a second exact reference implementation.

## Cases

### 1. Exact predictive quotient under Delta

The benchmark uses the SL-Bench `action_blindness` controlled finite world. Exact horizon-2 trace profiles define one distinguishability neuron per declared action probe:

```text
delta_probe(x,y) = TV(P(. | x, probe), P(. | y, probe))
```

Population zero-distance must recover the exact predictive quotient computed independently by the finite oracle.

For this world the expected quotient is:

```text
((0, 2), (1,), (3,))
```

This is a direct architecture check: with a sufficient distinction family, `Delta` must induce the same exact indistinguishability classes as the oracle.

### 2. Temporal identity under K

A transported identity is represented by the empirical successor kernel, not by static observation equality. Repeated `kitten -> young-cat` transport with a smaller competing successor must produce `young-cat` as the dominant continuation.

### 3. Intervention-sensitive C

Causal influence must remain zero after intervention evidence alone. A directional effect becomes available only after both intervention and control evidence exist.

The v0.1 scalar effect is intentionally simple; the invariant being tested is the separation of intervention evidence from ordinary temporal correlation.

### 4. Context-dependent active topology

The same persistent neuron population is exposed to two phenomena:

- a moving-object context, where the active higher-order relation should be `shape-motion`;
- a collision/causal context, where it should reorganize to `motion-cause`.

The neuron set is unchanged. Context, causal evidence and orientation change the active simplicial topology.

This checks the intended distinction:

```text
fixed potential substrate != active phenomenon-dependent topology
```

### 5. Contradiction-preserving integration

Two active local views that disagree beyond the configured tolerance must leave a visible integration residual and conflict coordinate rather than being silently treated as one coherent articulation.

## Acceptance contract

`research/five_operator_bench/run.py` emits a machine-readable report with five cases. v0.1 passes only if all five pass.

The focused preflight also runs:

- the original five-operator mechanism invariants;
- the new five-operator benchmark tests;
- the exact SL-Bench oracle tests.

## Non-claims

A v0.1 pass does not show that Cortex can discover the right neurons, topology, value axes or causal probes autonomously. It does not establish consciousness or subjective experience. It does not promote the mechanism into the frozen native runtime.

It establishes only that the current shared architecture is internally capable of carrying the required structures when those structures are supplied faithfully.

## Next stage

If v0.1 passes, the first autonomy experiment should replace oracle-supplied predictive distinction neurons with a learned candidate bank while retaining the exact finite oracle purely as evaluator.

The target becomes:

```text
stream -> learned local distinctions -> induced quotient
                                  |
                                  v
                         exact oracle audit
```

The primary metrics should be:

- quotient distortion;
- minimum feasible cell excess;
- held-out predictive loss;
- distinction count / active topology size;
- topology churn;
- recovery after controlled drift.

Only distinctions that earn their predictive or structural cost should survive.
