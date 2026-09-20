"""Cortex v1.16-R exact semantic-geometry fast-path campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from fused_stage_attribution_campaign import (
    ANCHOR_CASES,
    RESOLUTIONS,
    make_trace,
    run_case as stage_attribution_case,
)
from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import make_anchors
from support_sparse_prediction import (
    ExactSemanticGeometryPredictiveCache,
    FusedSupportSparseLocalResolutionController,
    SupportSparsePredictiveCache,
)

OUTCOMES = 3
DEVELOPMENT_SEEDS = (0, 1, 2)
OFFICIAL_SEEDS = (300, 301, 302)
PROTOCOL = "exact-semantic-geometry-fast-path-v1.16-R"


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
    *,
    fast: bool,
) -> FusedSupportSparseLocalResolutionController:
    field = _field(anchors.copy())
    controller = FusedSupportSparseLocalResolutionController(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
        loss_change_tolerance=0.001,
        full_scan_interval=64,
        complexity_refresh_interval=32,
        retained_mass=1.0,
    )
    if not fast:
        controller.cache = SupportSparsePredictiveCache(
            field,
            resolutions=RESOLUTIONS,
            feature_tolerance=0.0025,
        )
    elif not isinstance(
        controller.cache,
        ExactSemanticGeometryPredictiveCache,
    ):
        raise TypeError("fast controller did not install exact geometry cache")
    return controller


def _run_step(
    controller: FusedSupportSparseLocalResolutionController,
    observation: np.ndarray,
    outcome: int,
):
    step_start = perf_counter_ns()

    bundle = controller.prepare(observation)
    controller._update_tracker_from_weights(
        bundle.full_weights,
        outcome,
        bundle.candidate_predictions,
    )
    controller._update_field_from_weights(
        bundle.full_weights,
        outcome,
    )

    refresh_start = perf_counter_ns()
    dirty = controller.cache.refresh()
    refresh_ns = perf_counter_ns() - refresh_start
    step_ns = perf_counter_ns() - step_start

    return bundle, dirty, refresh_ns, step_ns


def run_pair_case(
    seed: int,
    anchors_count: int,
    *,
    equivalent_steps: int = 60,
    divergent_steps: int = 90,
    reconverged_steps: int = 120,
    include_stage_attribution: bool = True,
) -> dict:
    anchors = make_anchors(
        seed + 1709 + anchors_count,
        anchors=anchors_count,
        individual_scale=0.35,
    )
    trace = make_trace(
        seed + 5003,
        anchors,
        equivalent_steps=equivalent_steps,
        divergent_steps=divergent_steps,
        reconverged_steps=reconverged_steps,
    )

    legacy = _controller(anchors, fast=False)
    fast = _controller(anchors, fast=True)

    max_candidate_error = 0.0
    max_local_error = 0.0
    max_outcome_count_error = 0.0
    max_feature_error = 0.0
    max_probability_error = 0.0
    max_distance_error = 0.0
    max_kernel_error = 0.0
    dirty_disagreements = 0
    structural_disagreements = 0

    legacy_refresh_ns = 0
    fast_refresh_ns = 0
    legacy_step_ns = 0
    fast_step_ns = 0

    for index, (observation, outcome, _phase) in enumerate(trace):
        if index % 2 == 0:
            legacy_result = _run_step(
                legacy,
                observation,
                outcome,
            )
            fast_result = _run_step(
                fast,
                observation,
                outcome,
            )
        else:
            fast_result = _run_step(
                fast,
                observation,
                outcome,
            )
            legacy_result = _run_step(
                legacy,
                observation,
                outcome,
            )

        legacy_bundle, legacy_dirty, ref_ns, ref_step_ns = legacy_result
        fast_bundle, fast_dirty, opt_ns, opt_step_ns = fast_result
        legacy_refresh_ns += ref_ns
        fast_refresh_ns += opt_ns
        legacy_step_ns += ref_step_ns
        fast_step_ns += opt_step_ns

        max_candidate_error = max(
            max_candidate_error,
            float(
                np.max(
                    np.abs(
                        legacy_bundle.candidate_predictions
                        - fast_bundle.candidate_predictions
                    )
                )
            ),
        )
        max_local_error = max(
            max_local_error,
            float(
                np.max(
                    np.abs(
                        legacy_bundle.local_prediction
                        - fast_bundle.local_prediction
                    )
                )
            ),
        )
        max_outcome_count_error = max(
            max_outcome_count_error,
            float(
                np.max(
                    np.abs(
                        legacy.field.outcome_counts
                        - fast.field.outcome_counts
                    )
                )
            ),
        )
        max_feature_error = max(
            max_feature_error,
            float(
                np.max(
                    np.abs(
                        legacy.cache.features
                        - fast.cache.features
                    )
                )
            ),
        )
        max_probability_error = max(
            max_probability_error,
            float(
                np.max(
                    np.abs(
                        legacy.cache.probabilities
                        - fast.cache.probabilities
                    )
                )
            ),
        )
        max_distance_error = max(
            max_distance_error,
            float(
                np.max(
                    np.abs(
                        legacy.cache.distances
                        - fast.cache.distances
                    )
                )
            ),
        )
        max_kernel_error = max(
            max_kernel_error,
            float(
                np.max(
                    np.abs(
                        legacy.cache.kernels
                        - fast.cache.kernels
                    )
                )
            ),
        )

        if not np.array_equal(
            legacy_dirty,
            fast_dirty,
        ):
            dirty_disagreements += 1
        if not np.array_equal(
            legacy.current_indices,
            fast.current_indices,
        ):
            structural_disagreements += 1

    steps = len(trace)
    legacy_refresh_us = legacy_refresh_ns / steps / 1.0e3
    fast_refresh_us = fast_refresh_ns / steps / 1.0e3
    legacy_step_us = legacy_step_ns / steps / 1.0e3
    fast_step_us = fast_step_ns / steps / 1.0e3

    row = {
        "protocol": PROTOCOL,
        "seed": seed,
        "anchors": anchors_count,
        "steps": steps,
        "max_candidate_error": max_candidate_error,
        "max_local_error": max_local_error,
        "max_outcome_count_error": max_outcome_count_error,
        "max_feature_error": max_feature_error,
        "max_probability_error": max_probability_error,
        "max_distance_error": max_distance_error,
        "max_kernel_error": max_kernel_error,
        "dirty_disagreements": dirty_disagreements,
        "structural_disagreements": structural_disagreements,
        "legacy_refresh_us_per_step": legacy_refresh_us,
        "fast_refresh_us_per_step": fast_refresh_us,
        "refresh_speedup": (
            legacy_refresh_us
            / max(fast_refresh_us, 1.0e-15)
        ),
        "legacy_fused_us_per_step": legacy_step_us,
        "fast_fused_us_per_step": fast_step_us,
        "fused_speedup": (
            legacy_step_us
            / max(fast_step_us, 1.0e-15)
        ),
    }

    if include_stage_attribution:
        stage = stage_attribution_case(
            seed,
            anchors_count,
        )
        row.update(
            {
                "stage_total_us_per_step": (
                    stage["total_us_per_step"]
                ),
                "stage_accounted_fraction": (
                    stage["accounted_fraction"]
                ),
                "stage_cache_refresh_share": (
                    stage["cache_refresh_share"]
                ),
                "stage_candidate_readout_share": (
                    stage["candidate_readout_share"]
                ),
                "stage_hysteresis_scan_share": (
                    stage["hysteresis_scan_share"]
                ),
                "stage_complexity_share": (
                    stage["complexity_share"]
                ),
                "stage_perceptual_share": (
                    stage["perceptual_share"]
                ),
            }
        )

    return row


def _median(rows, key):
    return float(np.median([row[key] for row in rows]))


def campaign(
    seed_values: tuple[int, ...],
    output: Path,
) -> dict:
    rows = [
        run_pair_case(seed, anchors)
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
        cases[str(anchors)] = {
            "refresh_speedup": _median(
                selected,
                "refresh_speedup",
            ),
            "fused_speedup": _median(
                selected,
                "fused_speedup",
            ),
            "fast_refresh_us_per_step": _median(
                selected,
                "fast_refresh_us_per_step",
            ),
            "stage_total_us_per_step": _median(
                selected,
                "stage_total_us_per_step",
            ),
            "stage_accounted_fraction": _median(
                selected,
                "stage_accounted_fraction",
            ),
            "stage_cache_refresh_share": _median(
                selected,
                "stage_cache_refresh_share",
            ),
            "stage_candidate_readout_share": _median(
                selected,
                "stage_candidate_readout_share",
            ),
            "stage_hysteresis_scan_share": _median(
                selected,
                "stage_hysteresis_scan_share",
            ),
            "stage_complexity_share": _median(
                selected,
                "stage_complexity_share",
            ),
            "stage_perceptual_share": _median(
                selected,
                "stage_perceptual_share",
            ),
        }

    summary = {
        "protocol": PROTOCOL,
        "seeds": list(seed_values),
        "anchor_cases": list(ANCHOR_CASES),
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
        "max_feature_error": max(
            row["max_feature_error"]
            for row in rows
        ),
        "max_probability_error": max(
            row["max_probability_error"]
            for row in rows
        ),
        "max_distance_error": max(
            row["max_distance_error"]
            for row in rows
        ),
        "max_kernel_error": max(
            row["max_kernel_error"]
            for row in rows
        ),
        "dirty_disagreements": sum(
            row["dirty_disagreements"]
            for row in rows
        ),
        "structural_disagreements": sum(
            row["structural_disagreements"]
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
            "exact-semantic-geometry-fast-path-v1.16-R.jsonl"
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
