"""Validity checks for fused-stage-attribution-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STAGES = (
    "hysteresis_scan",
    "complexity",
    "perceptual",
    "candidate_readout",
    "local_readout",
    "tracker_update",
    "field_update",
    "cache_refresh",
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

    finite = all(
        np.isfinite(
            float(row["total_us_per_step"])
        )
        and all(
            np.isfinite(
                float(
                    row[
                        f"{stage}_us_per_step"
                    ]
                )
            )
            for stage in STAGES
        )
        for row in rows
    )
    coverage = all(
        0.90 <= row["accounted_fraction"] <= 1.0
        for row in rows
    )
    nonnegative = all(
        row[f"{stage}_share"] >= 0.0
        for row in rows
        for stage in STAGES
    )

    payload = {
        "protocol": "fused-stage-attribution-v1",
        "rows": len(rows),
        "validity": {
            "finite_timings": finite,
            "accounting_coverage": coverage,
            "nonnegative_shares": nonnegative,
        },
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
            "invalid fused-stage attribution: "
            + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
