"""Frozen gates for Cortex v1.22-R single-buffer kernel refresh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from single_buffer_kernel_refresh_campaign import (
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

    by_anchors = {
        anchors: [
            row for row in rows
            if int(row["anchors"]) == anchors
        ]
        for anchors in ANCHOR_CASES
    }

    performance = {
        anchors: {
            "refresh_speedup": float(
                np.median(
                    [
                        float(row["refresh_speedup"])
                        for row in selected
                    ]
                )
            ),
            "fused_speedup": float(
                np.median(
                    [
                        float(row["fused_speedup"])
                        for row in selected
                    ]
                )
            ),
            "fused_win_fraction": float(
                np.mean(
                    [
                        float(row["fused_speedup"]) > 1.0
                        for row in selected
                    ]
                )
            ),
        }
        for anchors, selected in by_anchors.items()
    }

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
                and key not in {
                    "seed",
                    "anchors",
                    "steps",
                    "reference_stale_before_access",
                    "candidate_stale_before_access",
                }
            )
        ),
        "candidate_prediction_parity": (
            metrics["max_candidate_error"] <= 1.0e-12
        ),
        "local_prediction_parity": (
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
        "no_hot_distance_materialization": all(
            int(row["reference_hot_distance_materializations"]) == 0
            and int(row["candidate_hot_distance_materializations"]) == 0
            for row in rows
        ),
        "lazy_distance_state_preserved": all(
            bool(row["reference_stale_before_access"])
            and bool(row["candidate_stale_before_access"])
            and int(
                row["reference_distance_materializations_after_access"]
            ) == 1
            and int(
                row["candidate_distance_materializations_after_access"]
            ) == 1
            for row in rows
        ),
        "single_buffer_shape": all(
            int(row["candidate_scratch_ndim"]) == 2
            for row in rows
        ),
        "single_buffer_exercised": all(
            int(row["candidate_single_buffer_refreshes"]) > 0
            for row in rows
        ),
        "small_scale_no_material_fused_regression": (
            performance[72]["fused_speedup"] >= 0.98
            and performance[144]["fused_speedup"] >= 0.98
        ),
        "large_scale_refresh_gain": (
            performance[288]["refresh_speedup"] >= 1.02
        ),
        "large_scale_fused_gain": (
            performance[288]["fused_speedup"] >= 1.01
        ),
        "large_scale_fused_seed_majority": (
            performance[288]["fused_win_fraction"] >= 0.60
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
        "performance": performance,
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
