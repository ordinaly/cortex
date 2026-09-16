# Cortex v1.0-RC1 Rust migration report

## Completed in this migration pass

- Froze Cortex Python v0.9.5-alpha as the executable specification.
- Recorded SHA-256 hashes for every frozen source/report file.
- Created a Cargo workspace with a native continual/plasticity core and a thin PyO3 Python extension.
- Ported in source the v0.9.5 hot-path semantics: recurrence/novelty evidence, bounded prototype memory, adaptive volatility, burst tensor plasticity, sampled curvature surveillance, prospective split validation, and merge reconciliation.
- Replaced Python object identity in split bookkeeping with stable native prototype IDs.
- Added a sparse relation/dependency crate to remove the dense O(N^2) representation as the intended v1 default.
- Generated four golden replay traces from the frozen Python engine: update, split, novelty, and split→merge.
- Added GitHub Actions CI for formatting, Clippy, core tests, native extension build, wheel build, and Rust↔Python golden replay.
- Added a Python research interface and native replay benchmark.
- Added a repository portability/freeze audit.

## Local validation completed

- Frozen reference checksum gate: PASS.
- Python source bytecode/import syntax check: PASS.
- Golden fixture generation from frozen reference: PASS.
- Golden fixture outcomes:
  - update: no split;
  - split: one promoted split;
  - novelty: second regime discovered;
  - merge: split promoted and later reconciled.
- Portability audit outside the frozen legacy snapshot: PASS.

## Native validation pending

The execution container contains no Rust toolchain and cannot resolve external package hosts. Therefore this pass could not honestly execute `cargo check`, `cargo test`, Maturin compilation, Clippy, or native golden replay.

Those steps are mandatory CI gates in `.github/workflows/ci.yml`; the Rust backend must not be designated release-default until they pass.

## Intended next action

Push this initialized repository to a new private GitHub repository, allow CI to compile the native core, fix any compile/differential discrepancies, then measure native latency and memory against the frozen Python reference. After phase-1 native parity, migrate the earlier entity/invariance/binding/relation runtime onto the sparse Rust graph store.
