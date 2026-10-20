# Probe Economy v0.3 — Results

## Outcome

**Pass.** The learner-visible exact probe economy found strictly smaller probe
populations in all three preregistered benchmark cases while preserving the full
empirical quotient and the independent finite-oracle result.

| Case | Declared probes | Minimum active probes | Reduction | Oracle distortion | Excess cells |
|---|---:|---:|---:|---:|---:|
| deterministic action blindness, H=2 | 6 | 1 | 83.3% | 0 | 0 |
| noisy immediate control, H=2 | 6 | 1 | 83.3% | 0 | 0 |
| noisy delayed control, H=3 | 14 | 2 | 85.7% | 0 | 0 |

The selected declared-probe indices were:

- deterministic action blindness: `[5]`;
- noisy immediate: `[5]`;
- noisy delayed: `[9, 13]`.

The index values are implementation-local positions in the declared probe order,
not semantic labels.

## Quotient preservation

### Deterministic action blindness, horizon 2

Both full and reduced empirical populations produced

```text
((0, 2), (1,), (3,))
```

using one active probe instead of six. The exact finite oracle reported three
minimum feasible cells, zero distortion, and no counterexample witnesses.

### Noisy immediate control, horizon 2

Both full and reduced empirical populations produced

```text
((0, 2), (1, 3))
```

using one active probe instead of six. The exact finite oracle reported two
minimum feasible cells, zero distortion, and no counterexample witnesses.

### Noisy delayed control, horizon 3

Both full and reduced empirical populations produced

```text
((0,), (1,), (2,), (3,))
```

using two active probes instead of fourteen. The exact finite oracle reported
four minimum feasible cells, zero distortion, and no counterexample witnesses.

This case is useful because the minimum is not one: the required distinction
relation genuinely needs at least two complementary learned questions at this
resolution.

## Dynamic reactivation

A separate finite-window test moved the useful one-step distinction from probe
`(0,)` to probe `(1,)`.

The controller first reduced the active population to `(0,)`. After the window
was flushed by the new regime, `(0,)` became insufficient; `(1,)` was
reactivated from latency and `(0,)` was retired. Full empirical distinction
coverage remained 1.0 throughout each accepted post-step state.

## Interpretation

This result supports a limited but important claim:

> Once a candidate question family is available, Cortex can identify a much
> smaller active basis of distinctions and can revise that basis when the
> phenomenon changes, without consulting hidden-world state.

It does **not** yet show that Cortex can discover the candidate questions or
allocate sampling effort economically. All declared probes were still trained.
The next experiment should therefore make question sampling itself active and
cost-sensitive while retaining periodic latent-probe checks for rediscovery.
