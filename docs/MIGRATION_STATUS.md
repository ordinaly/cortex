# Rust migration status

## Frozen baseline

Cortex v0.9.5-alpha has been frozen byte-for-byte as the Python executable specification. SHA-256 hashes are checked in CI.

Canonical release settings correct the known benchmark-harness drift by fixing `curvature_stride = 2`. The historical benchmark script remains preserved unchanged in the reference tree for provenance.

## Native core phase 1

Implemented in source:

- regime prototypes and bounded memory;
- evidence-gated recurrence and novelty;
- endogenous volatility/noise tracking;
- burst-gated low-rank tensor correction;
- conditional tensor refresh;
- sampled residual-curvature monitor with stride-adjusted EMA;
- rank-based split proposals;
- two-stage prospective split validation;
- one-level structural split;
- persistence-gated merge/reconciliation;
- explicit unresolved/budget-pressure state;
- serializable native snapshots.

## Semantic differences intentionally introduced by migration

1. Split candidates use stable `u64` prototype IDs rather than CPython object addresses. This changes representation, not intended semantics.
2. Distance evaluation iterates directly over contiguous Rust vectors; no NumPy carrier matrix is required.
3. Symmetric rank checks use the in-core Jacobi eigensolver. Floating operation order differs from NumPy `eigh`, so differential tests use numerical tolerances while requiring identical structural decisions.

## Validation status in this sandbox

The OS container does not have a Rust compiler and has no outbound package-network access, so native compilation cannot be executed here. The following have been validated locally:

- frozen reference hashes;
- import/bytecode validity of Python research/reference scripts;
- generation of four golden replay fixtures from the frozen Python engine;
- fixture outcomes: update stays unsplit, split promotes, novelty discovers, split→merge reconciles.

Native compilation, Clippy and Rust↔Python differential replay are encoded as mandatory GitHub Actions gates and remain **pending CI execution**.

## Next phases after phase-1 CI goes green

1. benchmark native phase-1 latency/memory against Python v0.9.5;
2. migrate sparse entity/relation/dependency state from v0.7;
3. remove dense `O(N^2)` relation carrier as the default representation;
4. rerun full v0.7–v0.9.5 regression suite;
5. external DRIFT/TGB validation under the frozen RC configuration.
