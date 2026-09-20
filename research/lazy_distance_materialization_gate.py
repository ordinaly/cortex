"""Validity gates for Cortex v1.20-R lazy distance materialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lazy_distance_materialization_campaign import (
    ANCHOR_CASES,
    DEVELOPMENT_SEEDS,
    OFFICIAL_SEEDS,
    PROTOCOL,
)


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

    metrics = {
        "max_candidate_error": max(
            float(row["max_candidate_error"])
            for row in rows
        ),
        "max_local_error": max(
            float(row["max_local_error"])
            for row in rows
        ),
        "max_outcome_count_error": max(
            float(row["max_outcome_count_error"])
            for row in rows
        ),
        "max_feature_error": max(
            float(row["max_feature_error"])
            for row in rows
        ),
        "max_probability_error": max(
            float(row["max_probability_error"])
            for row in rows
        ),
        "max_kernel_error": max(
            float(row["max_kernel_error"])
            for row in rows
        ),
        "max_distance_error_after_access": max(
            float(row["max_distance_error_after_access"])
            for row in rows
        ),
        "dirty_disagreements": sum(
            int(row["dirty_disagreements"])
            for row in rows
        ),
        "structural_disagreements": sum(
            int(row["structural_disagreements"])
            for row in rows
        ),
        "hot_distance_materializations": sum(
            int(row["hot_distance_materializations"])
            for row in rows
        ),
    }

    checks = {
        "protocol_exact": all(
            row["protocol"] == PROTOCOL
            for row in rows
        ),
        "row_count": (
            len(rows)
            == len(ANCHOR_CASES) * len(seeds)
        ),
        "anchor_cases_exact": sorted(
            {int(row["anchors"]) for row in rows}
        ) == sorted(ANCHOR_CASES),
        "seeds_exact": sorted(
            {int(row["seed"]) for row in rows}
        ) == sorted(seeds),
        "finite_metrics": all(
            np.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if (
                isinstance(value, (int, float))
                and key
                not in {
                    "seed",
                    "anchors",
                    "steps",
                    "stale_before_access",
                    "stale_after_access",
                }
            )
        ),
        "candidate_parity": (
            metrics["max_candidate_error"] <= 1.0e-12
        ),
        "local_parity": (
            metrics["max_local_error"] <= 1.0e-12
        ),
        "outcome_count_parity": (
            metrics["max_outcome_count_error"] <= 1.0e-12
        ),
        "feature_parity": (
            metrics["max_feature_error"] <= 1.0e-12
        ),
        "probability_parity": (
            metrics["max_probability_error"] <= 1.0e-12
        ),
        "kernel_parity": (
            metrics["max_kernel_error"] <= 1.0e-12
        ),
        "distance_view_parity": (
            metrics["max_distance_error_after_access"]
            <= 1.0e-12
        ),
        "dirty_parity": (
            metrics["dirty_disagreements"] == 0
        ),
        "structural_parity": (
            metrics["structural_disagreements"] == 0
        ),
        "no_hot_distance_materialization": (
            metrics["hot_distance_materializations"] == 0
        ),
        "distance_view_was_stale": all(
            bool(row["stale_before_access"])
            for row in rows
        ),
        "distance_view_fresh_after_access": all(
            not bool(row["stale_after_access"])
            for row in rows
        ),
        "one_materialization_on_access": all(
            int(
                row["distance_materializations_after_access"]
            ) == 1
            for row in rows
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
