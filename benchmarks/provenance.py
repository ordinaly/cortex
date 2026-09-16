"""Shared provenance fields for Cortex benchmark artifacts."""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def benchmark_provenance(protocol: str) -> dict[str, object]:
    from cortex import __spec_version__, __version__, native_version

    canonical = (ROOT / "VERSION").read_text(encoding="utf8").strip()
    spec = (ROOT / "SPEC_VERSION").read_text(encoding="utf8").strip()
    native = native_version()
    if native != canonical:
        raise RuntimeError(f"native version {native!r} does not match VERSION {canonical!r}")
    if __spec_version__ != spec:
        raise RuntimeError(
            f"Python spec version {__spec_version__!r} does not match SPEC_VERSION {spec!r}"
        )

    return {
        "benchmark_schema": 1,
        "benchmark_protocol": protocol,
        "cortex_version": canonical,
        "python_package_version": __version__,
        "native_version": native,
        "spec_version": spec,
        "git_sha": git_sha(),
        "python_runtime": platform.python_version(),
    }
