"""Frozen gates for exact-perceptual-fast-path-v1.14-R."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ANCHOR_CASES = (72, 144, 288)
SEEDS = (0, 1, 2)


def median(rows, key):
    return float(
        np.median(
            [row[key] for row in rows]
        )
    )


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

    rows_288 = [
        row for row in rows
        if row["anchors"] == 288
    ]

    metrics = {
        "max_weight_error": max(
            row["max_weight_error"]
            for row in rows
        ),
        "max_candidate_error": max(
            row["max_candidate_error"]
            for row in rows
        ),
        "max_local_error": max(
            row["max_local_error"]
            for row in rows
        ),
        "max_outcome_count_error": max(
            row["max_outcome_count_error"]
            for row in rows
        ),
        "structural_disagreements": sum(
            row["structural_disagreements"]
            for row in rows
        ),
        "perceptual_speedup_288": median(
            rows_288,
            "perceptual_speedup",
        ),
        "fused_speedup_288": median(
            rows_288,
            "fused_speedup",
        ),
        "stage_perceptual_share_288": median(
            rows_288,
            "stage_perceptual_share",
        ),
        "stage_accounting_min": min(
            row["stage_accounted_fraction"]
            for row in rows
        ),
    }

    checks = {
        "protocol_exact": all(
            row["protocol"]
            == "exact-perceptual-fast-path-v1.14-R"
            for row in rows
        ),
        "row_count": len(rows) == 9,
        "anchor_cases_exact": sorted(
            {row["anchors"] for row in rows}
        ) == list(ANCHOR_CASES),
        "seeds_exact": sorted(
            {row["seed"] for row in rows}
        ) == list(SEEDS),
        "finite_metrics": all(
            np.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if (
                isinstance(value, (int, float))
                and key not in {"seed", "anchors", "steps"}
            )
        ),
        "weight_parity": (
            metrics["max_weight_error"] <= 1.0e-12
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
        "structural_parity": (
            metrics["structural_disagreements"] == 0
        ),
        "perceptual_speedup_288": (
            metrics["perceptual_speedup_288"] >= 2.0
        ),
        "fused_speedup_288": (
            metrics["fused_speedup_288"] >= 1.15
        ),
        "stage_accounting": (
            metrics["stage_accounting_min"] >= 0.90
        ),
        "bottleneck_shift_288": (
            metrics["stage_perceptual_share_288"] <= 0.30
        ),
    }

    payload = {
        "protocol": "exact-perceptual-fast-path-v1.14-R",
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
