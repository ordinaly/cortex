"""Cortex v1.22-R single-buffer kernel-refresh campaign."""
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
)
from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import make_anchors
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
    LazyDistanceSemanticGeometryPredictiveCache,
    SingleBufferKernelRefreshPredictiveCache,
)

OUTCOMES = 3
PROTOCOL = "single-buffer-kernel-refresh-v1.22-R"
DEVELOPMENT_SEEDS = (30, 31, 32, 33, 34, 35)
OFFICIAL_SEEDS = (1200, 1201, 1202, 1203, 1204)


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
    candidate: bool,
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
    cache_type = (
        SingleBufferKernelRefreshPredictiveCache
        if candidate
        else LazyDistanceSemanticGeometryPredictiveCache
    )
    controller.cache = cache_type(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )
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


def run_case(seed: int, anchors_count: int) -> dict:
    anchors = make_anchors(
        seed + 1907 + anchors_count,
        anchors=anchors_count,
        individual_scale=0.35,
    )
    trace = make_trace(
        seed + 6007,
        anchors,
    )

    reference = _controller(anchors, candidate=False)
    candidate = _controller(anchors, candidate=True)
    assert isinstance(
        reference.cache,
        LazyDistanceSemanticGeometryPredictiveCache,
    )
    assert isinstance(
        candidate.cache,
        SingleBufferKernelRefreshPredictiveCache,
    )

    max_candidate_error = 0.0
    max_local_error = 0.0
    max_outcome_count_error = 0.0
    max_feature_error = 0.0
    max_probability_error = 0.0
    max_kernel_error = 0.0
    dirty_disagreements = 0
    structural_disagreements = 0

    reference_refresh_ns = 0
    candidate_refresh_ns = 0
    reference_step_ns = 0
    candidate_step_ns = 0

    for index, (observation, outcome, _phase) in enumerate(trace):
        if index % 2 == 0:
            reference_result = _run_step(
                reference,
                observation,
                outcome,
            )
            candidate_result = _run_step(
                candidate,
                observation,
                outcome,
            )
        else:
            candidate_result = _run_step(
                candidate,
                observation,
                outcome,
            )
            reference_result = _run_step(
                reference,
                observation,
                outcome,
            )

        (
            reference_bundle,
            reference_dirty,
            ref_refresh_ns,
            ref_step_ns,
        ) = reference_result
        (
            candidate_bundle,
            candidate_dirty,
            opt_refresh_ns,
            opt_step_ns,
        ) = candidate_result

        reference_refresh_ns += ref_refresh_ns
        candidate_refresh_ns += opt_refresh_ns
        reference_step_ns += ref_step_ns
        candidate_step_ns += opt_step_ns

        max_candidate_error = max(
            max_candidate_error,
            float(
                np.max(
                    np.abs(
                        reference_bundle.candidate_predictions
                        - candidate_bundle.candidate_predictions
                    )
                )
            ),
        )
        max_local_error = max(
            max_local_error,
            float(
                np.max(
                    np.abs(
                        reference_bundle.local_prediction
                        - candidate_bundle.local_prediction
                    )
                )
            ),
        )
        max_outcome_count_error = max(
            max_outcome_count_error,
            float(
                np.max(
                    np.abs(
                        reference.field.outcome_counts
                        - candidate.field.outcome_counts
                    )
                )
            ),
        )
        max_feature_error = max(
            max_feature_error,
            float(
                np.max(
                    np.abs(
                        reference.cache.features
                        - candidate.cache.features
                    )
                )
            ),
        )
        max_probability_error = max(
            max_probability_error,
            float(
                np.max(
                    np.abs(
                        reference.cache.probabilities
                        - candidate.cache.probabilities
                    )
                )
            ),
        )
        max_kernel_error = max(
            max_kernel_error,
            float(
                np.max(
                    np.abs(
                        reference.cache.kernels
                        - candidate.cache.kernels
                    )
                )
            ),
        )

        if not np.array_equal(
            reference_dirty,
            candidate_dirty,
        ):
            dirty_disagreements += 1
        if not np.array_equal(
            reference.current_indices,
            candidate.current_indices,
        ):
            structural_disagreements += 1

    reference_hot_materializations = (
        reference.cache.distance_materializations
    )
    candidate_hot_materializations = (
        candidate.cache.distance_materializations
    )
    reference_stale_before_access = (
        reference.cache.distance_view_stale
    )
    candidate_stale_before_access = (
        candidate.cache.distance_view_stale
    )

    reference_distances = reference.cache.distances.copy()
    candidate_distances = candidate.cache.distances.copy()
    max_distance_error = float(
        np.max(
            np.abs(
                reference_distances - candidate_distances
            )
        )
    )

    steps = len(trace)
    reference_refresh_us = (
        reference_refresh_ns / steps / 1.0e3
    )
    candidate_refresh_us = (
        candidate_refresh_ns / steps / 1.0e3
    )
    reference_step_us = (
        reference_step_ns / steps / 1.0e3
    )
    candidate_step_us = (
        candidate_step_ns / steps / 1.0e3
    )

    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "anchors": anchors_count,
        "steps": steps,
        "max_candidate_error": max_candidate_error,
        "max_local_error": max_local_error,
        "max_outcome_count_error": max_outcome_count_error,
        "max_feature_error": max_feature_error,
        "max_probability_error": max_probability_error,
        "max_kernel_error": max_kernel_error,
        "max_distance_error_after_access": max_distance_error,
        "dirty_disagreements": dirty_disagreements,
        "structural_disagreements": structural_disagreements,
        "reference_hot_distance_materializations": (
            reference_hot_materializations
        ),
        "candidate_hot_distance_materializations": (
            candidate_hot_materializations
        ),
        "reference_stale_before_access": (
            reference_stale_before_access
        ),
        "candidate_stale_before_access": (
            candidate_stale_before_access
        ),
        "reference_distance_materializations_after_access": (
            reference.cache.distance_materializations
        ),
        "candidate_distance_materializations_after_access": (
            candidate.cache.distance_materializations
        ),
        "candidate_scratch_ndim": (
            candidate.cache._kernel_scratch.ndim
        ),
        "candidate_scratch_capacity_rows": (
            candidate.cache.scratch_capacity_rows
        ),
        "candidate_scratch_bytes": (
            candidate.cache.scratch_bytes
        ),
        "candidate_scratch_reallocations": (
            candidate.cache.scratch_reallocations
        ),
        "candidate_single_buffer_refreshes": (
            candidate.cache.single_buffer_refreshes
        ),
        "reference_refresh_us_per_step": reference_refresh_us,
        "candidate_refresh_us_per_step": candidate_refresh_us,
        "refresh_speedup": (
            reference_refresh_us
            / max(candidate_refresh_us, 1.0e-15)
        ),
        "reference_fused_us_per_step": reference_step_us,
        "candidate_fused_us_per_step": candidate_step_us,
        "fused_speedup": (
            reference_step_us
            / max(candidate_step_us, 1.0e-15)
        ),
    }


def _median(rows, key):
    return float(
        np.median(
            [float(row[key]) for row in rows]
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
            if int(row["anchors"]) == anchors
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
            "candidate_refresh_us_per_step": _median(
                selected,
                "candidate_refresh_us_per_step",
            ),
            "candidate_fused_us_per_step": _median(
                selected,
                "candidate_fused_us_per_step",
            ),
            "fused_win_fraction": float(
                np.mean(
                    [
                        float(row["fused_speedup"]) > 1.0
                        for row in selected
                    ]
                )
            ),
            "refresh_win_fraction": float(
                np.mean(
                    [
                        float(row["refresh_speedup"]) > 1.0
                        for row in selected
                    ]
                )
            ),
            "max_scratch_bytes": max(
                int(row["candidate_scratch_bytes"])
                for row in selected
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
        "max_kernel_error": max(
            row["max_kernel_error"]
            for row in rows
        ),
        "max_distance_error_after_access": max(
            row["max_distance_error_after_access"]
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
            "single-buffer-kernel-refresh-v1.22-R.jsonl"
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
