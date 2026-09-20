"""Cortex v1.21-R post-lazy low-overhead stage profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hysteretic_local_resolution_campaign import make_anchors
from low_overhead_stage_profile_campaign import (
    ANCHOR_CASES,
    STAGES,
    _controller,
    run_case as v119_run_case,
)
from support_sparse_prediction import (
    LazyDistanceSemanticGeometryPredictiveCache,
)

PROTOCOL = "post-lazy-low-overhead-stage-profile-v1.21-R"
DEVELOPMENT_SEEDS = tuple(range(10, 16))
OFFICIAL_SEEDS = (1100, 1101, 1102, 1103, 1104)


def _verify_lazy_default() -> None:
    anchors = make_anchors(
        2121,
        anchors=18,
        individual_scale=0.35,
    )
    controller = _controller(anchors)
    if not isinstance(
        controller.cache,
        LazyDistanceSemanticGeometryPredictiveCache,
    ):
        raise TypeError(
            "v1.21-R requires the promoted v1.20-R lazy cache "
            "as the default fused research cache"
        )


def run_case(seed: int, anchors_count: int) -> dict:
    row = dict(v119_run_case(seed, anchors_count))
    row["protocol"] = PROTOCOL
    row["lazy_cache_default"] = True
    return row


def _median(rows, key):
    return float(np.median([row[key] for row in rows]))


def campaign(
    seed_values: tuple[int, ...],
    output: Path,
) -> dict:
    _verify_lazy_default()

    rows = [
        run_case(seed, anchors)
        for anchors in ANCHOR_CASES
        for seed in seed_values
    ]
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    cases = {}
    for anchors in ANCHOR_CASES:
        selected = [
            row
            for row in rows
            if row["anchors"] == anchors
        ]
        stages = {
            stage: {
                "us_per_step": _median(
                    selected,
                    f"{stage}_us_per_step",
                ),
                "share": _median(
                    selected,
                    f"{stage}_share",
                ),
            }
            for stage in STAGES
        }
        dominant = max(
            STAGES,
            key=lambda stage: stages[stage]["share"],
        )
        cases[str(anchors)] = {
            "total_us_per_step": _median(
                selected,
                "total_us_per_step",
            ),
            "accounted_fraction": _median(
                selected,
                "accounted_fraction",
            ),
            "dominant_stage": dominant,
            "stages": stages,
            "kernel_row_refresh_fraction": _median(
                selected,
                "kernel_row_refresh_fraction",
            ),
            "anchor_scan_fraction": _median(
                selected,
                "anchor_scan_fraction",
            ),
            "svd_fraction": _median(
                selected,
                "svd_fraction",
            ),
        }

    summary = {
        "protocol": PROTOCOL,
        "seeds": list(seed_values),
        "anchor_cases": list(ANCHOR_CASES),
        "lazy_cache_default": True,
        "max_checksum_error": max(
            row["checksum_error"]
            for row in rows
        ),
        "cases": cases,
    }
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--development", action="store_true")
    group.add_argument("--official", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "post-lazy-low-overhead-stage-profile-v1.21-R.jsonl"
        ),
    )
    args = parser.parse_args()
    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    campaign(seeds, args.output)


if __name__ == "__main__":
    main()
