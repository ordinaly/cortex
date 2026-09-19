"""Revised official gates for algebra-form-induction-v1.6.

The original v1 preflight low-coverage hypothesis for the subtraction control was
retired after unit characterization showed that a non-associative operation can
still satisfy other exact short identities. The v1.1 official campaign then exposed a second issue: per-partition witness minima suppressed true low-arity laws. v1.2 fixed that failure but admitted sparse-fit forms that did not generalize predictively. v1.3 added a single held-aside predictive split, which removed random-magma false closure but remained sparse and allowed adverse multi-law interactions. v1.4 used four-fold cross-fitted predictive evidence plus joint law-set validation, but still over-generalized locally supported dihedral identities beyond their witnessed element values. v1.5 added an empirical applicability basin per variable role, which fixed unsafe dihedral extrapolation but conflated structural form discovery with predictive applicability for low-arity laws. v1.6 separates structural membership from predictive activation: a law form may be discovered while remaining inapplicable to unsupported substitutions. All negative results are preserved.
"""
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

    def pattern_rate(family, name):
        return mean(
            family,
            lambda row: int(row["detected_patterns"][name]),
        )

    def metric(family, name):
        return mean(
            family,
            lambda row: row["metrics"]["cortex-fuzzy"][name],
        )

    checks = {
        "cyclic_rebracketing_discovered": pattern_rate(
            "cyclic-7", "rebracket"
        ) >= 0.95,
        "cyclic_swap_discovered": pattern_rate(
            "cyclic-7", "swap"
        ) >= 0.95,
        "dihedral_rebracketing_discovered": pattern_rate(
            "dihedral-4", "rebracket"
        ) >= 0.95,
        "dihedral_swap_rejected": pattern_rate(
            "dihedral-4", "swap"
        ) <= 0.05,
        "semilattice_repeat_discovered": pattern_rate(
            "semilattice-min-6", "repeat"
        ) >= 0.90,
        "left_zero_projection_discovered": pattern_rate(
            "left-zero-6", "left_projection"
        ) >= 0.95,
        "subtraction_rebracketing_rejected": pattern_rate(
            "subtraction-7", "rebracket"
        ) <= 0.05,
        "cyclic_precision": metric(
            "cyclic-7", "resolved_accuracy"
        ) >= 0.999,
        "cyclic_coverage": metric(
            "cyclic-7", "coverage"
        ) >= 0.85,
        "dihedral_precision": metric(
            "dihedral-4", "resolved_accuracy"
        ) >= 0.999,
        "dihedral_coverage": metric(
            "dihedral-4", "coverage"
        ) >= 0.75,
        "semilattice_precision": metric(
            "semilattice-min-6", "resolved_accuracy"
        ) >= 0.999,
        "left_zero_precision": metric(
            "left-zero-6", "resolved_accuracy"
        ) >= 0.999,
        "subtraction_conservative_precision": metric(
            "subtraction-7", "resolved_accuracy"
        ) >= 0.999,
        "random_magma_low_coverage": metric(
            "random-magma-6", "coverage"
        ) <= 0.10,
        "random_magma_no_rebracketing": pattern_rate(
            "random-magma-6", "rebracket"
        ) <= 0.05,
        "direct_memory_zero_coverage": max(
            row["metrics"]["direct-memory"]["coverage"]
            for row in rows
        ) == 0.0,
    }

    payload = {
        "protocol": "algebra-form-induction-v1.4",
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
