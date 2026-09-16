# Cortex v1.0-RC1 migration workspace

This repository is the Rust-core / Python-research-interface migration of Cortex.

**Author:** Arnold von Bauer-Gauss

## Status

- **Frozen executable specification:** Cortex Python v0.9.5-alpha under `reference/v0_9_5/`.
- **Native migration target:** `crates/cortex-core`.
- **Python extension:** `crates/cortex-python` via PyO3.
- **Research interface:** `python/cortex/`.
- **Differential oracle:** compressed golden traces under `tests/fixtures/`.
- **Canonical configuration:** `configs/v1_rc1.json`.

The native core is not allowed to become the default implementation until the four golden replay contracts pass in CI.

## Architecture boundary

Rust owns the stateful hot path:

1. bounded regime/prototype store;
2. recurrence and novelty evidence;
3. burst-gated tensor plasticity;
4. sampled residual-curvature refinement;
5. split/merge bookkeeping;
6. bounded-memory counters and structural events.

Python owns research orchestration:

1. dataset loading;
2. experiment configuration;
3. plotting/reporting;
4. baselines;
5. notebooks and analysis.

One call to `Cortex.step()` executes one complete observation update inside Rust; the API deliberately avoids fine-grained Python↔Rust crossings.

## Reference freeze

The exact files frozen from v0.9.5 are hashed in:

```text
reference/v0_9_5/SHA256SUMS.txt
```

Verify with:

```bash
pytest tests/test_freeze.py
```

Do not edit files in `reference/v0_9_5/`. A semantic change requires a new reference version.

## Build

Prerequisites:

- Rust 1.98.1 (pinned in `rust-toolchain.toml`)
- Python 3.11+
- Maturin 1.15.0

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip maturin==1.15.0 pytest numpy
pip install -e .
pytest
```

## Differential migration gate

The golden traces cover:

- local parametric adaptation;
- rank-triggered concept split;
- genuine novelty/discovery;
- split followed by reconciliation/merge.

The CI replay checks every step for prediction, current concept, structural event flags, memory pressure and prototype count, then checks final structural state within fixed numerical tolerances.

## Current scope

The repository migrates the v0.9.5 continual/plasticity hot core first. The earlier full Cortex entity/invariance/binding/relation stack remains a separate migration phase. This is deliberate: v0.9.5 is the executable specification and native kernels are introduced only behind differential contracts.

## License and attribution

Cortex is licensed under the **Apache License, Version 2.0** (`Apache-2.0`).

Original author and project creator: **Arnold von Bauer-Gauss**.

The repository includes a top-level `NOTICE` file containing the project attribution. Under Section 4(d) of Apache-2.0, redistributions of derivative works that include this work must preserve the applicable attribution notices from `NOTICE` in a readable form. See `LICENSE`, `NOTICE`, and `AUTHORS.md`.

This attribution requirement does **not** imply endorsement of derivative products by the author.
