"""Validity gates for Cortex v1.21-R post-lazy stage profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from low_overhead_stage_profile_campaign import (
    ANCHOR_CASES,
    STAGES,
)
from post_lazy_stage_profile_campaign import (
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

    checks = {
        "protocol_exact": all(
            row["protocol"] == PROTOCOL
            for row in rows
        ),
        "lazy_cache_default": all(
            bool(row["lazy_cache_default"])
            for row in rows
        ),
        "row_count": (
            len(rows)
            == len(ANCHOR_CASES) * len(seeds)
        ),
        "seed_set": sorted(
            {int(row["seed"]) for row in rows}
        ) == sorted(seeds),
        "anchor_cases": sorted(
            {int(row["anchors"]) for row in rows}
        ) == sorted(ANCHOR_CASES),
        "checksum_parity": all(
            float(row["checksum_error"]) <= 1.0e-12
            for row in rows
        ),
        "finite_timings": all(
            np.isfinite(float(row[key]))
            and float(row[key]) >= 0.0
            for row in rows
            for key in (
                ["total_us_per_step"]
                + [
                    f"{stage}_us_per_step"
                    for stage in STAGES
                ]
                + [
                    f"{stage}_share"
                    for stage in STAGES
                ]
            )
        ),
        "positive_total": all(
            float(row["total_us_per_step"]) > 0.0
            for row in rows
        ),
        "diagnostic_fractions": all(
            0.0 <= float(row[key]) <= 1.0
            for row in rows
            for key in (
                "kernel_row_refresh_fraction",
                "anchor_scan_fraction",
                "svd_fraction",
            )
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
