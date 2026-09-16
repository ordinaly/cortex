#!/usr/bin/env python3
"""Fail when Cortex release/specification versions drift across package surfaces."""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMVER_RE = re.compile(
    r"^(?P<base>\d+\.\d+\.\d+)(?:-(?P<kind>alpha|beta|rc)\.(?P<num>\d+))?$"
)
SPEC_RE = re.compile(r"^spec-v(?P<version>\d+\.\d+\.\d+)$")


def fail(message: str) -> None:
    print(f"version consistency error: {message}", file=sys.stderr)
    raise SystemExit(1)


def pep440_from_semver(version: str) -> str:
    match = SEMVER_RE.fullmatch(version)
    if not match:
        fail(f"VERSION is not supported SemVer: {version!r}")
    base = match.group("base")
    kind = match.group("kind")
    num = match.group("num")
    if kind is None:
        return base
    suffix = {"alpha": "a", "beta": "b", "rc": "rc"}[kind]
    return f"{base}{suffix}{num}"


def quoted_assignment(path: Path, name: str) -> str:
    text = path.read_text(encoding="utf8")
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']\s*$", text, re.M)
    if not match:
        fail(f"could not find {name} assignment in {path.relative_to(ROOT)}")
    return match.group(1)


def main() -> None:
    canonical = (ROOT / "VERSION").read_text(encoding="utf8").strip()
    pep440 = pep440_from_semver(canonical)
    spec = (ROOT / "SPEC_VERSION").read_text(encoding="utf8").strip()
    spec_match = SPEC_RE.fullmatch(spec)
    if not spec_match:
        fail(f"SPEC_VERSION has invalid form: {spec!r}")

    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf8"))
    cargo_version = cargo["workspace"]["package"]["version"]
    if cargo_version != canonical:
        fail(f"Cargo workspace version {cargo_version!r} != VERSION {canonical!r}")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf8"))
    python_version = pyproject["project"]["version"]
    if python_version != pep440:
        fail(f"pyproject version {python_version!r} != expected PEP 440 {pep440!r}")

    init_path = ROOT / "python" / "cortex" / "__init__.py"
    init_version = quoted_assignment(init_path, "__version__")
    if init_version != pep440:
        fail(f"cortex.__version__ literal {init_version!r} != expected {pep440!r}")
    init_spec = quoted_assignment(init_path, "__spec_version__")
    if init_spec != spec:
        fail(f"cortex.__spec_version__ literal {init_spec!r} != SPEC_VERSION {spec!r}")

    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf8"))
    cortex_packages = [
        pkg for pkg in lock.get("package", []) if str(pkg.get("name", "")).startswith("cortex-")
    ]
    if not cortex_packages:
        fail("Cargo.lock contains no cortex-* packages")
    drift = [(pkg["name"], pkg["version"]) for pkg in cortex_packages if pkg["version"] != canonical]
    if drift:
        fail(f"Cargo.lock Cortex package versions drift from {canonical!r}: {drift}")

    frozen_dir = ROOT / "reference" / ("v" + spec_match.group("version").replace(".", "_"))
    if not frozen_dir.is_dir():
        fail(f"frozen specification directory is missing: {frozen_dir.relative_to(ROOT)}")
    if not (frozen_dir / "SHA256SUMS.txt").is_file():
        fail(f"frozen specification hash manifest is missing under {frozen_dir.relative_to(ROOT)}")

    print(
        f"version consistency OK: Cortex {canonical} / Python {pep440} / specification {spec}"
    )


if __name__ == "__main__":
    main()
