"""Frozen gates for Cortex open-world-novelty-attribution-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_world_novelty_campaign import (
    DEVELOPMENT_SEEDS,
    FAMILIES,
    OFFICIAL_SEEDS,
    PROTOCOL,
)


def rate(rows, predicate) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return sum(
        int(predicate(row))
        for row in rows
    ) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        type=Path,
    )
    parser.add_argument(
        "--official",
        action="store_true",
    )
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
            row
            for row in rows
            if row["family"] == family
        ]
        for family in FAMILIES
    }
    resolved_rows = [
        row
        for row in rows
        if row["expected_category"]
        != "unresolved"
    ]
    persistent_rows = [
        row
        for row in rows
        if row["model"][
            "persistent_edit_expected"
        ]
    ]
    conservative_rows = [
        row
        for row in rows
        if not row["model"][
            "persistent_edit_expected"
        ]
    ]

    metrics = {
        "attribution_accuracy": rate(
            rows,
            lambda row: row["checks"][
                "category_correct"
            ],
        ),
        "resolution_correctness": rate(
            rows,
            lambda row: row["checks"][
                "resolved_correct"
            ],
        ),
        "pure_evaluation_rate": rate(
            rows,
            lambda row: row["model"][
                "pure_evaluation"
            ],
        ),
        "edit_locality_rate": rate(
            rows,
            lambda row: row["model"][
                "edit_locality"
            ],
        ),
        "required_edit_rate": rate(
            persistent_rows,
            lambda row: row["checks"][
                "required_edit_present"
            ],
        ),
        "post_edit_stability_rate": rate(
            persistent_rows,
            lambda row: row["checks"][
                "post_edit_stable"
            ],
        ),
        "false_persistent_edit_rate": (
            1.0
            - rate(
                conservative_rows,
                lambda row: row["checks"][
                    "no_false_edit"
                ],
            )
        ),
        "resolved_margin_rate": rate(
            resolved_rows,
            lambda row: (
                row["read"]["margin"]
                >= 0.5
            ),
        ),
        "ambiguous_unresolved_rate": rate(
            by_family[
                "ambiguous-instance-type"
            ],
            lambda row: (
                row["checks"][
                    "ambiguous_no_edit"
                ]
                and row["checks"][
                    "ambiguous_competitors"
                ]
            ),
        ),
        "transient_recovery_rate": rate(
            by_family[
                "transient-excursion"
            ],
            lambda row: row["checks"][
                "transient_recovery"
            ],
        ),
        "nearest_baseline_accuracy": rate(
            rows,
            lambda row: row["baselines"][
                "nearest_correct"
            ],
        ),
        "reset_baseline_accuracy": rate(
            rows,
            lambda row: row["baselines"][
                "reset_correct"
            ],
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
        "attribution_accuracy": (
            metrics[
                "attribution_accuracy"
            ] == 1.0
        ),
        "resolution_correctness": (
            metrics[
                "resolution_correctness"
            ] == 1.0
        ),
        "pure_evaluation": (
            metrics[
                "pure_evaluation_rate"
            ] == 1.0
        ),
        "edit_locality": (
            metrics[
                "edit_locality_rate"
            ] == 1.0
        ),
        "required_edits": (
            metrics[
                "required_edit_rate"
            ] == 1.0
        ),
        "post_edit_stability": (
            metrics[
                "post_edit_stability_rate"
            ] == 1.0
        ),
        "no_false_persistent_edits": (
            metrics[
                "false_persistent_edit_rate"
            ] == 0.0
        ),
        "resolved_margin": (
            metrics[
                "resolved_margin_rate"
            ] == 1.0
        ),
        "ambiguous_unresolved": (
            metrics[
                "ambiguous_unresolved_rate"
            ] == 1.0
        ),
        "transient_recovery": (
            metrics[
                "transient_recovery_rate"
            ] == 1.0
        ),
        "nearest_baseline_discriminated": (
            metrics[
                "nearest_baseline_accuracy"
            ] <= 0.50
        ),
        "global_reset_discriminated": (
            metrics[
                "reset_baseline_accuracy"
            ] <= 0.25
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
    print(
        json.dumps(
            payload,
            sort_keys=True,
        )
    )

    if not payload["passed"]:
        raise SystemExit(
            "failed gates: "
            + ", ".join(
                name
                for name, passed
                in checks.items()
                if not passed
            )
        )


if __name__ == "__main__":
    main()
