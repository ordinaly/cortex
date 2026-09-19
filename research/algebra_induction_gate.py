"""Validation gates for algebra-induction-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("no campaign rows")

    by_family = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    def mean(family, fn):
        values = [fn(row) for row in by_family[family]]
        return sum(values) / len(values)

    checks = {
        "cyclic_law_detection": mean(
            "cyclic-7", lambda row: int(row["law_detection_exact"])
        ) >= 0.95,
        "dihedral_law_detection": mean(
            "dihedral-4", lambda row: int(row["law_detection_exact"])
        ) >= 0.95,
        "control_law_rejection": mean(
            "subtraction-7", lambda row: int(row["law_detection_exact"])
        ) >= 0.95,
        "cyclic_heldout_precision": mean(
            "cyclic-7",
            lambda row: row["metrics"]["cortex"]["resolved_accuracy"],
        ) >= 0.999,
        "cyclic_heldout_coverage": mean(
            "cyclic-7",
            lambda row: row["metrics"]["cortex"]["coverage"],
        ) >= 0.95,
        "dihedral_heldout_precision": mean(
            "dihedral-4",
            lambda row: row["metrics"]["cortex"]["resolved_accuracy"],
        ) >= 0.999,
        "dihedral_heldout_coverage": mean(
            "dihedral-4",
            lambda row: row["metrics"]["cortex"]["coverage"],
        ) >= 0.98,
        "nonassoc_control_conservative": mean(
            "subtraction-7",
            lambda row: row["metrics"]["cortex"]["coverage"],
        ) <= 0.05,
        "no_closure_conflicts": max(row["conflicts"] for row in rows) == 0,
        "direct_memory_has_no_heldout_coverage": max(
            row["metrics"]["direct-memory"]["coverage"]
            for row in rows
        ) == 0.0,
    }

    payload = {
        "protocol": "algebra-induction-v1",
        "rows": len(rows),
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(payload, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(
            "failed gates: "
            + ", ".join(name for name, ok in checks.items() if not ok)
        )


if __name__ == "__main__":
    main()
