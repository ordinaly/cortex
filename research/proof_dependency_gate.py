"""Frozen gates for Cortex explicit-proof-dependency-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from proof_dependency_campaign import (
    DEVELOPMENT_SEEDS,
    FAMILIES,
    OFFICIAL_SEEDS,
    PROTOCOL,
)


def rate(rows, predicate) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(int(predicate(row)) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--official", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.results.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("no campaign rows")

    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    by_family = {
        family: [
            row for row in rows
            if row["family"] == family
        ]
        for family in FAMILIES
    }

    metrics = {
        "dag_valid_rate": rate(
            rows,
            lambda row: row["dag_valid"]
            and row["restored_dag_valid"],
        ),
        "provenance_valid_rate": rate(
            rows,
            lambda row: row["provenance_valid"],
        ),
        "selected_shortest_rate": rate(
            rows,
            lambda row: row["selected_is_shortest"],
        ),
        "restoration_exact_rate": rate(
            rows,
            lambda row: row["restoration_exact"],
        ),
        "redundant_survival_rate": rate(
            by_family["redundant-branches"],
            lambda row: (
                row["initial_state"] == 1
                and row["initial_positive_proofs"] >= 2
                and row["independent_pair_present"]
                and row["alternate_survives"]
                and row["after_removal_state"] == 1
                and row["exact_invalidation"]
            ),
        ),
        "single_proof_baseline_failure_rate": rate(
            (
                *by_family["redundant-branches"],
                *by_family["short-vs-long"],
            ),
            lambda row: (
                row["after_removal_state"] == 1
                and row["single_proof_baseline_state"] == 0
            ),
        ),
        "deeper_takeover_rate": rate(
            by_family["short-vs-long"],
            lambda row: (
                row["initial_state"] == 1
                and row["longer_proof_present"]
                and row["deeper_takeover"]
                and row["after_removal_state"] == 1
                and row["exact_invalidation"]
            ),
        ),
        "shared_root_invalidation_rate": rate(
            by_family["shared-root"],
            lambda row: (
                row["initial_state"] == 1
                and row["initial_positive_proofs"] >= 2
                and row["positive_proofs_share_root"]
                and row["after_shared_removal_state"] == 0
                and row["after_shared_removal_reason"]
                == "unproven"
                and row["after_shared_positive_proofs"] == 0
                and row["exact_invalidation"]
            ),
        ),
        "conflict_detection_rate": rate(
            by_family["contradictory"],
            lambda row: (
                row["initial_state"] == 0
                and row["initial_reason"] == "conflict"
                and row["initial_positive_proofs"] >= 1
                and row["initial_negative_proofs"] >= 1
                and row["without_negative_state"] == 1
                and row["without_negative_reason"] == "supported"
                and row["without_positive_state"] == -1
                and row["without_positive_reason"] == "refuted"
                and row["restored_conflict_state"] == 0
                and row["restored_conflict_reason"] == "conflict"
            ),
        ),
        "nested_partial_invalidation_rate": rate(
            by_family["nested-redundancy"],
            lambda row: (
                row["initial_state"] == 1
                and row["initial_positive_proofs"] >= 3
                and row["after_nested_removal_state"] == 1
                and row["proof_count_decreased"]
                and row["multiple_proofs_remain"]
                and row["exact_invalidation"]
            ),
        ),
        "irrelevant_locality_rate": rate(
            by_family["irrelevant-premise"],
            lambda row: (
                row["initial_state"] == 1
                and row["after_irrelevant_state"] == 1
                and row["target_fingerprint_unchanged"]
            ),
        ),
    }

    checks = {
        "protocol_exact": all(
            row["protocol"] == PROTOCOL
            for row in rows
        ),
        "row_count": (
            len(rows)
            == len(FAMILIES) * len(seeds)
        ),
        "families_exact": sorted(
            {row["family"] for row in rows}
        ) == sorted(FAMILIES),
        "seeds_exact": sorted(
            {int(row["seed"]) for row in rows}
        ) == list(seeds),
        "family_balance": all(
            len(selected) == len(seeds)
            for selected in by_family.values()
        ),
        "dag_valid": metrics["dag_valid_rate"] == 1.0,
        "provenance_valid": (
            metrics["provenance_valid_rate"] == 1.0
        ),
        "selected_shortest": (
            metrics["selected_shortest_rate"] == 1.0
        ),
        "restoration_exact": (
            metrics["restoration_exact_rate"] == 1.0
        ),
        "redundant_support": (
            metrics["redundant_survival_rate"] == 1.0
        ),
        "single_proof_baseline_discriminated": (
            metrics[
                "single_proof_baseline_failure_rate"
            ] == 1.0
        ),
        "shortest_to_longer_takeover": (
            metrics["deeper_takeover_rate"] == 1.0
        ),
        "shared_dependency_invalidation": (
            metrics[
                "shared_root_invalidation_rate"
            ] == 1.0
        ),
        "contradictory_proofs": (
            metrics["conflict_detection_rate"] == 1.0
        ),
        "nested_partial_invalidation": (
            metrics[
                "nested_partial_invalidation_rate"
            ] == 1.0
        ),
        "irrelevant_locality": (
            metrics["irrelevant_locality_rate"] == 1.0
        ),
    }

    payload = {
        "protocol": PROTOCOL,
        "mode": (
            "official"
            if args.official
            else "development"
        ),
        "rows": len(rows),
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(payload, sort_keys=True))

    if not payload["passed"]:
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
