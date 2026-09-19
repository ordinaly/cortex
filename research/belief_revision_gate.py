"""Frozen validity gates for belief-revision-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OFFICIAL_SEEDS = tuple(range(200, 220))
PRIOR_STRENGTHS = (1, 2, 3)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


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

    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)

    retraction_latencies = [
        row["strong_contradiction"]["first_null_batch"]
        for row in rows
        if row["strong_contradiction"]["first_null_batch"] is not None
    ]
    recovery_latencies = [
        row["recovery"]["active_batch"]
        for row in rows
        if row["recovery"]["active_batch"] is not None
    ]

    retraction_order = []
    recovery_order = []
    for seed in OFFICIAL_SEEDS:
        seed_rows = sorted(
            by_seed.get(seed, []),
            key=lambda row: row["prior_batches"],
        )
        if len(seed_rows) != 3:
            retraction_order.append(False)
            recovery_order.append(False)
            continue

        retract = [
            row["strong_contradiction"]["first_null_batch"]
            for row in seed_rows
        ]
        recover = [
            row["recovery"]["active_batch"]
            for row in seed_rows
        ]
        retraction_order.append(
            None not in retract
            and retract[0] < retract[1] < retract[2]
        )
        recovery_order.append(
            None not in recover
            and recover[0] < recover[1] < recover[2]
        )

    metrics = {
        "initial_active_rate": mean(
            int(row["initial"]["state"] == 1)
            for row in rows
        ),
        "single_batch_retention_rate": mean(
            int(row["single_batch"]["retained_active"])
            for row in rows
        ),
        "retraction_success_rate": mean(
            int(row["strong_contradiction"]["retraction_success"])
            for row in rows
        ),
        "contradiction_transition_order_rate": mean(
            int(
                row["strong_contradiction"]["first_unresolved_batch"]
                is not None
                and row["strong_contradiction"]["first_null_batch"]
                is not None
                and row["strong_contradiction"]["first_unresolved_batch"]
                < row["strong_contradiction"]["first_null_batch"]
            )
            for row in rows
        ),
        "max_retraction_latency": max(
            retraction_latencies,
            default=0,
        ),
        "retraction_prior_order_rate": mean(
            int(value)
            for value in retraction_order
        ),
        "ambiguous_unresolved_rate": mean(
            int(row["ambiguous"]["unresolved"])
            for row in rows
        ),
        "ambiguous_false_retraction_rate": mean(
            int(row["ambiguous"]["false_retraction"])
            for row in rows
        ),
        "recovery_success_rate": mean(
            int(row["recovery"]["success"])
            for row in rows
        ),
        "recovery_transition_order_rate": mean(
            int(
                row["recovery"]["first_unresolved_batch"]
                is not None
                and row["recovery"]["active_batch"]
                is not None
                and row["recovery"]["first_unresolved_batch"]
                < row["recovery"]["active_batch"]
            )
            for row in rows
        ),
        "max_recovery_latency": max(
            recovery_latencies,
            default=0,
        ),
        "recovery_prior_order_rate": mean(
            int(value)
            for value in recovery_order
        ),
        "downstream_alignment_rate": mean(
            int(row["downstream_alignment"])
            for row in rows
        ),
        "contradiction_margin_monotone_rate": mean(
            int(row["strong_contradiction"]["margin_monotone"])
            for row in rows
        ),
        "recovery_margin_monotone_rate": mean(
            int(row["recovery"]["margin_monotone"])
            for row in rows
        ),
        "unrelated_structure_preservation_rate": mean(
            int(row["unrelated_structure_preserved"])
            for row in rows
        ),
        "sticky_baseline_retraction_success": mean(
            int(row["sticky_baseline_retraction_success"])
            for row in rows
        ),
        "latest_batch_premature_retraction_rate": mean(
            int(row["latest_batch_premature_retraction"])
            for row in rows
        ),
    }

    checks = {
        "protocol_exact": all(
            row["protocol"] == "belief-revision-v1"
            for row in rows
        ),
        "row_count": len(rows) == 60,
        "official_seeds_exact": sorted(
            {row["seed"] for row in rows}
        ) == list(OFFICIAL_SEEDS),
        "prior_strengths_exact": sorted(
            {row["prior_batches"] for row in rows}
        ) == list(PRIOR_STRENGTHS),
        "initial_active_rate": (
            metrics["initial_active_rate"] == 1.0
        ),
        "single_batch_retention_rate": (
            metrics["single_batch_retention_rate"] == 1.0
        ),
        "retraction_success_rate": (
            metrics["retraction_success_rate"] == 1.0
        ),
        "contradiction_transition_order": (
            metrics["contradiction_transition_order_rate"] == 1.0
        ),
        "max_retraction_latency": (
            metrics["max_retraction_latency"] <= 22
        ),
        "retraction_prior_order": (
            metrics["retraction_prior_order_rate"] == 1.0
        ),
        "ambiguous_unresolved_rate": (
            metrics["ambiguous_unresolved_rate"] == 1.0
        ),
        "ambiguous_false_retraction_rate": (
            metrics["ambiguous_false_retraction_rate"] == 0.0
        ),
        "recovery_success_rate": (
            metrics["recovery_success_rate"] == 1.0
        ),
        "recovery_transition_order": (
            metrics["recovery_transition_order_rate"] == 1.0
        ),
        "max_recovery_latency": (
            metrics["max_recovery_latency"] <= 8
        ),
        "recovery_prior_order": (
            metrics["recovery_prior_order_rate"] == 1.0
        ),
        "downstream_alignment_rate": (
            metrics["downstream_alignment_rate"] == 1.0
        ),
        "contradiction_margin_monotone_rate": (
            metrics["contradiction_margin_monotone_rate"] == 1.0
        ),
        "recovery_margin_monotone_rate": (
            metrics["recovery_margin_monotone_rate"] == 1.0
        ),
        "unrelated_structure_preservation_rate": (
            metrics["unrelated_structure_preservation_rate"] == 1.0
        ),
        "sticky_baseline_retraction_success": (
            metrics["sticky_baseline_retraction_success"] == 0.0
        ),
        "latest_batch_premature_retraction_rate": (
            metrics["latest_batch_premature_retraction_rate"] == 1.0
        ),
    }

    payload = {
        "protocol": "belief-revision-v1",
        "rows": len(rows),
        "metrics": metrics,
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
