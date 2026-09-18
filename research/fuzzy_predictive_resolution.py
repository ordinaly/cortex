"""Resolution sweep for Cortex v1.6-R predictive sufficiency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fuzzy_predictive_field_campaign import run_case


RESOLUTIONS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50)


def campaign(
    seeds: int,
    output: Path,
    *,
    prediction_tolerance: float = 0.03,
) -> None:
    rows = []
    for resolution in RESOLUTIONS:
        for seed in range(seeds):
            row = run_case(
                seed,
                semantic_resolution=resolution,
            )
            row["protocol"] = "fuzzy-predictive-resolution-v1"
            rows.append(row)

    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    summaries = []
    for resolution in RESOLUTIONS:
        subset = [
            row for row in rows
            if row["semantic_resolution"] == resolution
        ]
        summaries.append(
            {
                "resolution": resolution,
                "hit_at_1": float(
                    np.mean(
                        [row["fuzzy_future_hit_at_1"] for row in subset]
                    )
                ),
                "effective_rank": float(
                    np.mean(
                        [row["predictive_effective_rank"] for row in subset]
                    )
                ),
            }
        )

    best_hit = max(row["hit_at_1"] for row in summaries)
    sufficient = [
        row for row in summaries
        if row["hit_at_1"] >= best_hit - prediction_tolerance
    ]
    selected = max(sufficient, key=lambda row: row["resolution"])

    for row in summaries:
        print(json.dumps({"suite": "resolution", **row}, sort_keys=True))
    print(
        json.dumps(
            {
                "suite": "selected",
                "prediction_tolerance": prediction_tolerance,
                "best_hit_at_1": best_hit,
                "selected_resolution": selected["resolution"],
                "selected_hit_at_1": selected["hit_at_1"],
                "selected_effective_rank": selected["effective_rank"],
                "physical_anchors": 24,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fuzzy-predictive-resolution-v1.jsonl"),
    )
    parser.add_argument(
        "--prediction-tolerance",
        type=float,
        default=0.03,
    )
    args = parser.parse_args()
    campaign(
        args.seeds,
        args.output,
        prediction_tolerance=args.prediction_tolerance,
    )


if __name__ == "__main__":
    main()
