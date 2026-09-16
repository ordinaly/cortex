from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "v0_9_5"


def test_reference_hashes_are_frozen():
    lines = (REF / "SHA256SUMS.txt").read_text().splitlines()
    assert lines
    for line in lines:
        digest, rel = line.split("  ", 1)
        path = ROOT / rel
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == digest, path
