"""Cortex v1.20-R lazy distance-materialization campaign."""
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
    ExactSemanticGeometryPredictiveCache,
    FusedSupportSparseLocalResolutionController,
    LazyDistanceSemanticGeometryPredictiveCache,
)

OUTCOMES = 3
PROTOCOL = "lazy-distance-materialization-v1.20-R"
DEVELOPMENT_SEEDS = tuple(range(8))
OFFICIAL_SEEDS = (900, 901, 902, 903, 904)


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
    lazy: bool,
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
        LazyDistanceSemanticGeometryPredictiveCache
        if lazy
        else ExactSemanticGeometryPredictiveCache
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
        seed + 1709 + anchors_count,
        anchors=anchors_count,
        individual_scale=0.35,
    )
    trace = make_trace(
        seed + 5003,
        anchors,
    )

    eager = _controller(anchors, lazy=False)
    lazy = _controller(anchors, lazy=True)
    assert isinstance(
        lazy.cache,
        LazyDistanceSemanticGeometryPredictiveCache,
    )

    max_candidate_error = 0.0
    max_local_error = 0.0
    max_outcome_count_error = 0.0
    max_feature_error = 0.0
    max_probability_error = 0.0
    max_kernel_error = 0.0
    dirty_disagreements = 0
    structural_disagreements = 0

    eager_refresh_ns = 0
    lazy_refresh_ns = 0
    eager_step_ns = 0
    lazy_step_ns = 0

    for index, (observation, outcome, _phase) in enumerate(trace):
        if index % 2 == 0:
            eager_result = _run_step(
                eager,
                observation,
                outcome,
            )
            lazy_result = _run_step(
                lazy,
                observation,
                outcome,
            )
        else:
            lazy_result = _run_step(
                lazy,
                observation,
                outcome,
            )
            eager_result = _run_step(
                eager,
                observation,
                outcome,
            )

        eager_bundle, eager_dirty, ref_ns, ref_step_ns = eager_result
        lazy_bundle, lazy_dirty, opt_ns, opt_step_ns = lazy_result
        eager_refresh_ns += ref_ns
        lazy_refresh_ns += opt_ns
        eager_step_ns += ref_step_ns
        lazy_step_ns += opt_step_ns

        max_candidate_error = max(
            max_candidate_error,
            float(
                np.max(
                    np.abs(
                        eager_bundle.candidate_predictions
                        - lazy_bundle.candidate_predictions
                    )
                )
            ),
        )
        max_local_error = max(
            max_local_error,
            float(
                np.max(
                    np.abs(
                        eager_bundle.local_prediction
                        - lazy_bundle.local_prediction
                    )
                )
            ),
        )
        max_outcome_count_error = max(
            max_outcome_count_error,
            float(
                np.max(
                    np.abs(
                        eager.field.outcome_counts
                        - lazy.field.outcome_counts
                    )
                )
            ),
        )
        max_feature_error = max(
            max_feature_error,
            float(
                np.max(
                    np.abs(
                        eager.cache.features
                        - lazy.cache.features
                    )
                )
            ),
        )
        max_probability_error = max(
            max_probability_error,
            float(
                np.max(
                    np.abs(
                        eager.cache.probabilities
                        - lazy.cache.probabilities
                    )
                )
            ),
        )
        max_kernel_error = max(
            max_kernel_error,
            float(
                np.max(
                    np.abs(
                        eager.cache.kernels
                        - lazy.cache.kernels
                    )
                )
            ),
        )

        if not np.array_equal(
            eager_dirty,
            lazy_dirty,
        ):
            dirty_disagreements += 1
        if not np.array_equal(
            eager.current_indices,
            lazy.current_indices,
        ):
            structural_disagreements += 1

    hot_materializations = lazy.cache.distance_materializations
    stale_before_access = lazy.cache.distance_view_stale

    access_start = perf_counter_ns()
    lazy_distances = lazy.cache.distances.copy()
    distance_access_ns = perf_counter_ns() - access_start
    eager_distances = eager.cache.distances
    max_distance_error = float(
        np.max(
            np.abs(
                eager_distances - lazy_distances
            )
        )
    )
    stale_after_access = lazy.cache.distance_view_stale

    steps = len(trace)
    eager_refresh_us = eager_refresh_ns / steps / 1.0e3
    lazy_refresh_us = lazy_refresh_ns / steps / 1.0e3
    eager_step_us = eager_step_ns / steps / 1.0e3
    lazy_step_us = lazy_step_ns / steps / 1.0e3

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
        "hot_distance_materializations": hot_materializations,
        "stale_before_access": stale_before_access,
        "stale_after_access": stale_after_access,
        "distance_materializations_after_access": (
            lazy.cache.distance_materializations
        ),
        "distance_access_us": (
            distance_access_ns / 1.0e3
        ),
        "eager_refresh_us_per_step": eager_refresh_us,
        "lazy_refresh_us_per_step": lazy_refresh_us,
        "refresh_speedup": (
            eager_refresh_us
            / max(lazy_refresh_us, 1.0e-15)
        ),
        "eager_fused_us_per_step": eager_step_us,
        "lazy_fused_us_per_step": lazy_step_us,
        "fused_speedup": (
            eager_step_us
            / max(lazy_step_us, 1.0e-15)
        ),
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
        cases[str(anchors)] = {
            "refresh_speedup": _median(
                selected,
                "refresh_speedup",
            ),
            "fused_speedup": _median(
                selected,
                "fused_speedup",
            ),
            "lazy_refresh_us_per_step": _median(
                selected,
                "lazy_refresh_us_per_step",
            ),
            "distance_access_us": _median(
                selected,
                "distance_access_us",
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
        "hot_distance_materializations": sum(
            row["hot_distance_materializations"]
            for row in rows
        ),
        "all_stale_before_access": all(
            row["stale_before_access"]
            for row in rows
        ),
        "all_fresh_after_access": all(
            not row["stale_after_access"]
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
            "lazy-distance-materialization-v1.20-R.jsonl"
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
