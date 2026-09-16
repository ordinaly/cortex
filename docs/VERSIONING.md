# Cortex versioning and provenance

Cortex uses a SemVer-compatible release scheme for the software implementation and keeps executable specifications and benchmark protocols on independent version tracks.

## Public Cortex release

The canonical release version lives in the repository root `VERSION` file and has the form:

```text
MAJOR.MINOR.PATCH[-PRERELEASE]
```

Examples:

```text
1.0.0
1.0.0-rc.2
1.1.0-beta.1
2.0.0-alpha.1
```

Git tags add the conventional `v` prefix, for example `v1.0.0-rc.2`.

### Meaning of the components

- **PATCH**: bug fixes, performance work, documentation, internal refactors, and other changes that preserve the published behavioral and public API contract.
- **MINOR**: backward-compatible capabilities or extensions to the published contract.
- **MAJOR**: deliberate incompatibilities or fundamental changes to the architecture or public behavioral contract.
- **Prerelease number**: identifies successive alpha, beta, or release-candidate snapshots before the corresponding stable release.

While Cortex is in a release-candidate series, prerelease increments may collect compatible engineering and validation work that would otherwise qualify as patch- or minor-level changes after `1.0.0` is stable.

## Frozen executable specification

The reference implementation is versioned independently from the software release. Its canonical identifier lives in `SPEC_VERSION` and has the form:

```text
spec-vMAJOR.MINOR.PATCH
```

The current frozen executable specification is:

```text
spec-v0.9.5
```

Its files live under `reference/v0_9_5/` and are protected by `SHA256SUMS.txt`. A semantic change to the executable specification requires a new specification version; it must not silently mutate an existing freeze.

## Python package normalization

The root `VERSION` file uses SemVer syntax. Python packaging uses the equivalent PEP 440 spelling when a prerelease is present:

```text
SemVer             PEP 440
1.0.0              1.0.0
1.0.0-rc.2         1.0.0rc2
1.1.0-beta.1       1.1.0b1
2.0.0-alpha.1      2.0.0a1
```

`pyproject.toml` and `cortex.__version__` therefore intentionally use the normalized PEP 440 form. CI checks that the two forms correspond to the same canonical release.

## Benchmark and experiment protocols

Scientific protocols are versioned independently of Cortex releases. They use descriptive identifiers such as:

```text
full-runtime-v1
native-scaling-v1
stage-attribution-v1
regime-saturation-v1
external-validation-v1
```

Changing a benchmark in a way that changes its workload, measurement semantics, aggregation, or acceptance criteria requires a new protocol version. Re-running the same protocol on a later Cortex release does not change the protocol identifier.

Every committed or published benchmark result should record at least:

- Cortex canonical SemVer release;
- Python/packaging version;
- native Rust package version;
- frozen executable specification identifier;
- benchmark protocol identifier;
- Git commit SHA;
- deterministic seed(s) and repetitions;
- relevant runtime/configuration parameters.

This keeps statements such as the following unambiguous:

> Results obtained with Cortex `v1.2.0`, executable specification `spec-v0.9.5`, using protocol `external-validation-v2`.

## Repository consistency gate

`scripts/check_version_consistency.py` verifies the canonical version against:

- root `Cargo.toml` workspace version;
- all Cortex package entries in `Cargo.lock`;
- `pyproject.toml`;
- `python/cortex/__init__.py`;
- the frozen specification identifier exposed by Python;
- the existence of the corresponding frozen reference tree.

The check is part of CI. Version changes should therefore begin by editing `VERSION` and then synchronizing the derived package forms in the same pull request.

## Release checklist

Before tagging a public release or prerelease:

1. update `VERSION` and all derived package versions;
2. run the version-consistency gate;
3. pass the Rust, Python differential, and wheel CI suites;
4. record any release-defining benchmark protocol/results with full provenance;
5. update README/release notes to the canonical version spelling;
6. create the matching Git tag, such as `v1.0.0-rc.2`.

The software release version, specification version, and experiment protocol are related provenance coordinates, not aliases for one another.
