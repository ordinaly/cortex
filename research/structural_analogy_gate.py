"""Frozen validity gates for structural-analogy-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OFFICIAL_SEEDS = tuple(range(300, 320))
FAMILIES = (
    "asymmetric-chain",
    "asymmetric-branch",
    "symmetric-twins",
    "symmetric-diamond",
)
ASYMMETRIC = {
    "asymmetric-chain",
    "asymmetric-branch",
}
SYMMETRIC = {
    "symmetric-twins",
    "symmetric-diamond",
}
BUDGETS = (12, 24, 36)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
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

    compatible = [
        row for row in rows
        if row.get("kind") == "compatible"
    ]
    full = [
        row for row in rows
        if row.get("kind") == "full-identifiability"
    ]
    near = [
        row for row in rows
        if row.get("kind") == "near-isomorphic"
    ]
    broken = [
        row for row in rows
        if row.get("kind") == "broken-analogy"
    ]

    resolved = sum(
        row["metrics"]["cortex"]["resolved"]
        for row in compatible
    )
    correct = sum(
        row["metrics"]["cortex"]["correct"]
        for row in compatible
    )

    positive_resolved = sum(
        row["metrics"]["cortex"]["positive_resolved"]
        for row in compatible
    )
    positive_correct = sum(
        row["metrics"]["cortex"]["positive_resolved"]
        for row in compatible
        if row["metrics"]["cortex"][
            "positive_resolved_accuracy"
        ] == 1.0
    )
    negative_resolved = sum(
        row["metrics"]["cortex"]["negative_resolved"]
        for row in compatible
    )
    negative_correct = sum(
        row["metrics"]["cortex"]["negative_resolved"]
        for row in compatible
        if row["metrics"]["cortex"][
            "negative_resolved_accuracy"
        ] == 1.0
    )

    coverage = {
        budget: mean(
            row["metrics"]["cortex"]["coverage"]
            for row in compatible
            if row["budget"] == budget
        )
        for budget in BUDGETS
    }

    monotone = []
    for seed in OFFICIAL_SEEDS:
        for family in FAMILIES:
            selected = sorted(
                (
                    row
                    for row in compatible
                    if row["seed"] == seed
                    and row["family"] == family
                ),
                key=lambda row: row["budget"],
            )
            counts = [
                row["candidate_count"]
                for row in selected
            ]
            monotone.append(
                len(counts) == len(BUDGETS)
                and all(
                    right <= left
                    for left, right in zip(
                        counts,
                        counts[1:],
                    )
                )
            )

    asym_full = [
        row for row in full
        if row["family"] in ASYMMETRIC
    ]
    sym_full = [
        row for row in full
        if row["family"] in SYMMETRIC
    ]

    asym_resolved = sum(
        row["mapping"]["entity_resolved"]
        + row["mapping"]["relation_resolved"]
        for row in compatible
        if row["family"] in ASYMMETRIC
    )
    asym_correct = sum(
        row["mapping"]["entity_correct"]
        + row["mapping"]["relation_correct"]
        for row in compatible
        if row["family"] in ASYMMETRIC
    )

    metrics = {
        "entity_label_overlap_max": max(
            row["entity_label_overlap"]
            for row in compatible
        ),
        "relation_label_overlap_max": max(
            row["relation_label_overlap"]
            for row in compatible
        ),
        "heldout_resolved_accuracy": (
            correct / max(1, resolved)
        ),
        "positive_resolved_accuracy": (
            positive_correct / max(1, positive_resolved)
        ),
        "negative_resolved_accuracy": (
            negative_correct / max(1, negative_resolved)
        ),
        "coverage_12": coverage[12],
        "coverage_24": coverage[24],
        "coverage_36": coverage[36],
        "direct_memory_coverage": mean(
            row["metrics"]["direct-memory"]["coverage"]
            for row in compatible
        ),
        "candidate_monotonicity_rate": mean(
            int(value)
            for value in monotone
        ),
        "asymmetric_full_unique_rate": mean(
            int(row["mapping_unique"])
            for row in asym_full
        ),
        "symmetric_full_ambiguity_rate": mean(
            int(
                (not row["mapping_unique"])
                and row["entity_ambiguous"] > 0
            )
            for row in sym_full
        ),
        "asymmetric_resolved_mapping_precision": (
            asym_correct / max(1, asym_resolved)
        ),
        "near_isomorphic_rejection_rate": mean(
            int(row["rejected"])
            for row in near
        ),
        "broken_analogy_rejection_rate": mean(
            int(
                row["contradiction_found"]
                and row["rejected"]
            )
            for row in broken
        ),
    }

    checks = {
        "protocol_exact": all(
            row.get("protocol") == "structural-analogy-v1"
            for row in rows
        ),
        "compatible_row_count": len(compatible) == 240,
        "full_row_count": len(full) == 80,
        "near_row_count": len(near) == 40,
        "broken_row_count": len(broken) == 80,
        "official_seeds_exact": sorted(
            {row["seed"] for row in compatible}
        ) == list(OFFICIAL_SEEDS),
        "families_exact": sorted(
            {row["family"] for row in compatible}
        ) == sorted(FAMILIES),
        "budgets_exact": sorted(
            {row["budget"] for row in compatible}
        ) == list(BUDGETS),
        "compatible_version_space_nonempty": all(
            row["candidate_count"] > 0
            for row in compatible
        ),
        "entity_label_overlap": (
            metrics["entity_label_overlap_max"] == 0
        ),
        "relation_label_overlap": (
            metrics["relation_label_overlap_max"] == 0
        ),
        "heldout_resolved_accuracy": (
            metrics["heldout_resolved_accuracy"] == 1.0
        ),
        "positive_resolved_accuracy": (
            metrics["positive_resolved_accuracy"] == 1.0
        ),
        "negative_resolved_accuracy": (
            metrics["negative_resolved_accuracy"] == 1.0
        ),
        "coverage_12": metrics["coverage_12"] >= 0.20,
        "coverage_24": metrics["coverage_24"] >= 0.45,
        "coverage_36": metrics["coverage_36"] >= 0.65,
        "direct_memory_coverage": (
            metrics["direct_memory_coverage"] == 0.0
        ),
        "candidate_monotonicity": (
            metrics["candidate_monotonicity_rate"] == 1.0
        ),
        "asymmetric_full_unique": (
            metrics["asymmetric_full_unique_rate"] == 1.0
        ),
        "asymmetric_mapping_precision": (
            metrics[
                "asymmetric_resolved_mapping_precision"
            ] == 1.0
        ),
        "symmetric_ambiguity_preserved": (
            metrics[
                "symmetric_full_ambiguity_rate"
            ] == 1.0
        ),
        "near_isomorphic_rejection": (
            metrics[
                "near_isomorphic_rejection_rate"
            ] == 1.0
        ),
        "broken_analogy_rejection": (
            metrics[
                "broken_analogy_rejection_rate"
            ] == 1.0
        ),
    }

    payload = {
        "protocol": "structural-analogy-v1",
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
