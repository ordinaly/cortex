"""Cortex v1.19-R low-overhead fused-stage crossover campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from fused_stage_attribution_campaign import (
    ANCHOR_CASES,
    RESOLUTIONS,
    candidate_from_weights,
    local_from_weights,
    make_trace,
)
from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import make_anchors
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
)

OUTCOMES = 3
PROTOCOL = "low-overhead-fused-stage-profile-v1.19-R"
DEVELOPMENT_SEEDS = tuple(range(6))
OFFICIAL_SEEDS = (600, 601, 602, 603, 604)

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


def _field(anchors: np.ndarray) -> FuzzyPredictiveField:
    return FuzzyPredictiveField(
        anchors,
        outcomes=OUTCOMES,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
        evidence_decay=0.998,
    )


def _controller(
    anchors: np.ndarray,
) -> FusedSupportSparseLocalResolutionController:
    return FusedSupportSparseLocalResolutionController(
        _field(anchors.copy()),
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
        loss_change_tolerance=0.001,
        full_scan_interval=64,
        complexity_refresh_interval=32,
        retained_mass=1.0,
    )


def _run_trace(
    seed: int,
    anchors_count: int,
    *,
    timed_stage: str | None,
) -> dict:
    anchors = make_anchors(
        seed + 1709 + anchors_count,
        anchors=anchors_count,
        individual_scale=0.35,
    )
    trace = make_trace(
        seed + 5003,
        anchors,
    )
    controller = _controller(anchors)

    timed_ns = 0
    total_ns = 0
    prediction_checksum = np.zeros(
        OUTCOMES,
        dtype=float,
    )

    def call(stage: str, fn):
        nonlocal timed_ns
        if timed_stage != stage:
            return fn()
        start = perf_counter_ns()
        value = fn()
        timed_ns += perf_counter_ns() - start
        return value

    for observation, outcome, _phase in trace:
        step_start = (
            perf_counter_ns()
            if timed_stage is None
            else None
        )

        call(
            "hysteresis_scan",
            controller._advance_state,
        )

        complexity = call(
            "complexity",
            controller._complexity,
        )
        if not np.isfinite(complexity):
            raise ValueError("non-finite complexity")

        weights = call(
            "perceptual",
            lambda: controller.field.perceptual_weights(
                observation
            ),
        )

        candidates = call(
            "candidate_readout",
            lambda: candidate_from_weights(
                controller,
                weights,
            ),
        )

        local_prediction = call(
            "local_readout",
            lambda: local_from_weights(
                controller,
                weights,
            ),
        )

        call(
            "tracker_update",
            lambda: controller._update_tracker_from_weights(
                weights,
                outcome,
                candidates,
            ),
        )

        call(
            "field_update",
            lambda: controller._update_field_from_weights(
                weights,
                outcome,
            ),
        )

        call(
            "cache_refresh",
            controller.cache.refresh,
        )

        prediction_checksum += local_prediction

        if step_start is not None:
            total_ns += perf_counter_ns() - step_start

    steps = len(trace)
    return {
        "timed_stage": (
            "total"
            if timed_stage is None
            else timed_stage
        ),
        "us_per_step": (
            total_ns / steps / 1.0e3
            if timed_stage is None
            else timed_ns / steps / 1.0e3
        ),
        "prediction_checksum": prediction_checksum.tolist(),
        "kernel_row_refresh_fraction": (
            controller.optimization_diagnostics()[
                "kernel_row_refresh_fraction"
            ]
        ),
        "anchor_scan_fraction": (
            controller.optimization_diagnostics()[
                "anchor_scan_fraction"
            ]
        ),
        "svd_fraction": (
            controller.optimization_diagnostics()[
                "svd_fraction"
            ]
        ),
    }


def run_case(seed: int, anchors_count: int) -> dict:
    # Rotate stage order by seed so transient runner load is not confounded
    # with one fixed stage ordering.
    offset = seed % len(STAGES)
    order = (
        STAGES[offset:]
        + STAGES[:offset]
    )

    stage_runs = {
        stage: _run_trace(
            seed,
            anchors_count,
            timed_stage=stage,
        )
        for stage in order
    }
    total_run = _run_trace(
        seed,
        anchors_count,
        timed_stage=None,
    )

    reference_checksum = np.asarray(
        total_run["prediction_checksum"],
        dtype=float,
    )
    checksum_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(
                        stage_runs[stage][
                            "prediction_checksum"
                        ],
                        dtype=float,
                    )
                    - reference_checksum
                )
            )
        )
        for stage in STAGES
    )

    total_us = total_run["us_per_step"]
    stage_us = {
        stage: stage_runs[stage]["us_per_step"]
        for stage in STAGES
    }
    shares = {
        stage: stage_us[stage] / max(total_us, 1.0e-15)
        for stage in STAGES
    }

    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "anchors": anchors_count,
        "steps": 270,
        "total_us_per_step": total_us,
        "checksum_error": checksum_error,
        "accounted_fraction": (
            sum(stage_us.values())
            / max(total_us, 1.0e-15)
        ),
        **{
            f"{stage}_us_per_step": stage_us[stage]
            for stage in STAGES
        },
        **{
            f"{stage}_share": shares[stage]
            for stage in STAGES
        },
        "kernel_row_refresh_fraction": total_run[
            "kernel_row_refresh_fraction"
        ],
        "anchor_scan_fraction": total_run[
            "anchor_scan_fraction"
        ],
        "svd_fraction": total_run["svd_fraction"],
    }


def _median(rows, key):
    return float(
        np.median(
            [row[key] for row in rows]
        )
    )


def campaign(
    seed_values: tuple[int, ...],
    output: Path,
) -> dict:
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
            row for row in rows
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
            "low-overhead-fused-stage-profile-v1.19-R.jsonl"
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
