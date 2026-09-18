"""Validation and tradeoff report for support-mass-tradeoff-v1."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def row_admissible(row: dict) -> bool:
    return (
        row["mean_local_tv"] <= 0.01
        and row["max_local_tv"] <= 0.05
        and row["mean_candidate_tv"] <= 0.01
        and row["max_candidate_tv"] <= 0.05
        and row["resolution_disagreement"] <= 0.05
        and row["compressed_divergent_nll"]
        <= row["baseline_divergent_nll"] + 0.02
        and row["recoarsened_fraction"] >= 0.95
        and row["max_dropped_mass"]
        <= 1.0 - row["retained_mass"] + 1.0e-12
    )


def evaluate(rows: list[dict]) -> dict:
    finite_keys = (
        "baseline_us_per_step",
        "compressed_us_per_step",
        "speedup_vs_full",
        "mean_local_tv",
        "max_local_tv",
        "mean_candidate_tv",
        "max_candidate_tv",
        "resolution_disagreement",
        "baseline_divergent_nll",
        "compressed_divergent_nll",
        "recoarsened_fraction",
        "mean_support_fraction",
        "max_support_fraction",
        "mean_dropped_mass",
        "max_dropped_mass",
    )

    finite = all(
        all(
            np.isfinite(float(row[key]))
            for key in finite_keys
        )
        for row in rows
    )
    mass_contract = all(
        row["max_dropped_mass"]
        <= 1.0 - row["retained_mass"] + 1.0e-12
        for row in rows
    )

    grouped: dict[tuple[int, float], list[dict]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                int(row["anchors"]),
                float(row["retained_mass"]),
            )
        ].append(row)

    characterization = {}
    for (anchors, mass), selected in sorted(
        grouped.items()
    ):
        characterization[
            f"{anchors}:{mass}"
        ] = {
            "admissible_seed_fraction": float(
                np.mean(
                    [
                        row_admissible(row)
                        for row in selected
                    ]
                )
            ),
            "median_speedup": float(
                np.median(
                    [
                        row["speedup_vs_full"]
                        for row in selected
                    ]
                )
            ),
            "median_support_fraction": float(
                np.median(
                    [
                        row["mean_support_fraction"]
                        for row in selected
                    ]
                )
            ),
        }

    def median(
        anchors: int,
        mass: float,
        key: str,
    ) -> float:
        selected = grouped[(anchors, mass)]
        return float(
            np.median(
                [row[key] for row in selected]
            )
        )

    hypotheses = {
        "q_099_support_fraction_288": (
            median(
                288,
                0.99,
                "mean_support_fraction",
            )
            <= 0.70
        ),
        "q_099_speedup_288": (
            median(
                288,
                0.99,
                "speedup_vs_full",
            )
            >= 1.10
        ),
        "q_0995_speedup_288": (
            median(
                288,
                0.995,
                "speedup_vs_full",
            )
            >= 1.05
        ),
    }

    return {
        "validity": {
            "finite_metrics": finite,
            "retained_mass_contract": mass_contract,
        },
        "characterization": characterization,
        "hypotheses": hypotheses,
    }


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

    evaluation = evaluate(rows)
    valid = all(
        evaluation["validity"].values()
    )
    payload = {
        "protocol": "support-mass-tradeoff-v1",
        "rows": len(rows),
        **evaluation,
        "valid": valid,
        "failed_hypotheses": [
            name
            for name, passed
            in evaluation["hypotheses"].items()
            if not passed
        ],
    }
    print(json.dumps(payload, sort_keys=True))

    if not valid:
        failed = [
            name
            for name, passed
            in evaluation["validity"].items()
            if not passed
        ]
        raise SystemExit(
            "invalid support-mass campaign: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
