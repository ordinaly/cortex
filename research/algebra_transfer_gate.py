"""Frozen official gates for algebra-transfer-v1.6.

Do not modify these thresholds after official campaign results are observed.
The protocol is documented in docs/ALGEBRA_TRANSFER_V1_PROTOCOL.md.
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

    compatible = [
        row for row in rows
        if row["kind"] == "compatible"
    ]
    random_controls = [
        row for row in rows
        if row["kind"] == "random-control"
    ]
    subtraction_controls = [
        row for row in rows
        if row["kind"] == "subtraction-control"
    ]

    def by_fraction(fraction: float):
        return [
            row for row in compatible
            if row["observed_fraction"] == fraction
        ]

    def mean(rows_, fn):
        values = [fn(row) for row in rows_]
        return sum(values) / len(values)

    low = by_fraction(0.20)
    mid = by_fraction(0.30)

    checks = {
        "fresh_target_seed_range": min(
            row["seed"] for row in rows
        ) >= 100,
        "source_target_labels_disjoint": max(
            row["source_target_label_overlap"]
            for row in rows
        ) == 0,
        "source_target_orders_disjoint": all(
            row["target_order"] not in row["source_orders"]
            for row in compatible
        ),
        "compatible_zero_wrong_resolved": max(
            row["metrics"]["transfer"]["wrong_resolved"]
            for row in compatible
        ) == 0,
        "compatible_zero_closure_conflicts": max(
            row["transfer_conflicts"]
            for row in compatible
        ) == 0,
        "low_evidence_mean_coverage": mean(
            low,
            lambda row: row["metrics"]["transfer"]["coverage"],
        ) >= 0.20,
        "low_evidence_transfer_gain": mean(
            low,
            lambda row: row["coverage_gain"],
        ) >= 0.10,
        "mid_evidence_transfer_gain": mean(
            mid,
            lambda row: row["coverage_gain"],
        ) >= 0.05,
        "transfer_library_bounded": max(
            row["library_size"]
            for row in rows
        ) <= 24,
        "transfer_candidate_reduction": max(
            row["library_size"]
            for row in compatible
        ) < min(
            row["scratch_grammar_forms"]
            for row in compatible
        ),
        "multiplicity_evidence_respected": all(
            row["transfer_selected"] == 0
            or (
                row["transfer_validation_positive"]
                >= row["transfer_required_predictions"]
                and row["transfer_validation_negative"] == 0
                and row["transfer_validation_conflicts"] == 0
            )
            for row in rows
        ),
        "random_controls_zero_wrong": max(
            row["metrics"]["transfer"]["wrong_resolved"]
            for row in random_controls
        ) == 0,
        "random_controls_abstain": max(
            row["metrics"]["transfer"]["resolved"]
            for row in random_controls
        ) == 0,
        "subtraction_zero_wrong": max(
            row["metrics"]["transfer"]["wrong_resolved"]
            for row in subtraction_controls
        ) == 0,
        "subtraction_rebracket_rejected": mean(
            subtraction_controls,
            lambda row: int(row["rebracket_selected"]),
        ) <= 0.05,
        "direct_memory_zero_coverage": max(
            row["metrics"]["direct-memory"]["coverage"]
            for row in compatible
        ) == 0.0,
    }

    payload = {
        "protocol": "algebra-transfer-v1.6",
        "rows": len(rows),
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(payload, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(
            "failed gates: "
            + ", ".join(
                name
                for name, passed in checks.items()
                if not passed
            )
        )


if __name__ == "__main__":
    main()
