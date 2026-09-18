"""Validation and performance report for normalized-anchor-perception-v1."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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

    parity = all(
        row["perception_checksum_error"] <= 1.0e-10
        and row["max_candidate_error"] <= 1.0e-10
        and row["max_local_error"] <= 1.0e-10
        and row["resolution_disagreement"] == 0.0
        and row["outcome_count_error"] <= 1.0e-10
        and row["loss_numerator_error"] <= 1.0e-10
        and row["anchor_mass_error"] <= 1.0e-10
        for row in rows
    )
    finite = all(
        np.isfinite(float(row["overall_speedup"]))
        and np.isfinite(float(row["perception_speedup"]))
        and row["overall_speedup"] > 0
        and row["perception_speedup"] > 0
        for row in rows
    )

    by_anchor: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_anchor[int(row["anchors"])].append(row)

    def median(anchors: int, key: str) -> float:
        return float(
            np.median(
                [row[key] for row in by_anchor[anchors]]
            )
        )

    hypotheses = {
        "perception_speedup_288": (
            median(
                288,
                "perception_speedup",
            )
            >= 5.0
        ),
        "overall_speedup_288": (
            median(
                288,
                "overall_speedup",
            )
            >= 1.30
        ),
        "overall_speedup_576": (
            median(
                576,
                "overall_speedup",
            )
            >= 1.30
        ),
    }

    payload = {
        "protocol": "normalized-anchor-perception-v1",
        "rows": len(rows),
        "validity": {
            "differential_parity": parity,
            "finite_positive_timings": finite,
        },
        "hypotheses": hypotheses,
        "failed_hypotheses": [
            name
            for name, passed in hypotheses.items()
            if not passed
        ],
    }
    payload["valid"] = all(
        payload["validity"].values()
    )
    print(json.dumps(payload, sort_keys=True))

    if not payload["valid"]:
        failed = [
            name
            for name, passed
            in payload["validity"].items()
            if not passed
        ]
        raise SystemExit(
            "invalid normalized-anchor campaign: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
