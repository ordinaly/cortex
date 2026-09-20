"""Cortex v1.18-R exact kernel-ladder fast-path campaign."""
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
    KernelLadderSemanticGeometryPredictiveCache,
)

OUTCOMES = 3
PROTOCOL = "kernel-ladder-fast-path-v1.18-R"
DEVELOPMENT_SEEDS = tuple(range(8))
OFFICIAL_SEEDS = (500, 501, 502)


class TimedExactSemanticGeometryPredictiveCache(
    ExactSemanticGeometryPredictiveCache
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_compute_ns = 0

    def _kernel_rows_from_squared(
        self,
        distance_squared: np.ndarray,
    ) -> np.ndarray:
        start = perf_counter_ns()
        value = super()._kernel_rows_from_squared(
            distance_squared
        )
        self.kernel_compute_ns += perf_counter_ns() - start
        return value


class TimedKernelLadderSemanticGeometryPredictiveCache(
    KernelLadderSemanticGeometryPredictiveCache
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_compute_ns = 0

    def _kernel_rows_from_squared(
        self,
        distance_squared: np.ndarray,
    ) -> np.ndarray:
        start = perf_counter_ns()
        value = super()._kernel_rows_from_squared(
            distance_squared
        )
        self.kernel_compute_ns += perf_counter_ns() - start
        return value


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
    ladder: bool,
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
        TimedKernelLadderSemanticGeometryPredictiveCache
        if ladder
        else TimedExactSemanticGeometryPredictiveCache
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


def run_case(
    seed: int,
    anchors_count: int,
    *,
    equivalent_steps: int = 60,
    divergent_steps: int = 90,
    reconverged_steps: int = 120,
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

    reference = _controller(anchors, ladder=False)
    ladder = _controller(anchors, ladder=True)

    max_candidate_error = 0.0
    max_local_error = 0.0
    max_outcome_count_error = 0.0
    max_feature_error = 0.0
    max_probability_error = 0.0
    max_distance_error = 0.0
    max_kernel_error = 0.0
    underflow_mismatches = 0
    dirty_disagreements = 0
    structural_disagreements = 0

    reference_refresh_ns = 0
    ladder_refresh_ns = 0
    reference_step_ns = 0
    ladder_step_ns = 0

    for index, (observation, outcome, _phase) in enumerate(trace):
        if index % 2 == 0:
            ref_result = _run_step(
                reference,
                observation,
                outcome,
            )
            ladder_result = _run_step(
                ladder,
                observation,
                outcome,
            )
        else:
            ladder_result = _run_step(
                ladder,
                observation,
                outcome,
            )
            ref_result = _run_step(
                reference,
                observation,
                outcome,
            )

        (
            ref_bundle,
            ref_dirty,
            ref_refresh,
            ref_step,
        ) = ref_result
        (
            ladder_bundle,
            ladder_dirty,
            ladder_refresh,
            ladder_step,
        ) = ladder_result

        reference_refresh_ns += ref_refresh
        ladder_refresh_ns += ladder_refresh
        reference_step_ns += ref_step
        ladder_step_ns += ladder_step

        max_candidate_error = max(
            max_candidate_error,
            float(
                np.max(
                    np.abs(
                        ref_bundle.candidate_predictions
                        - ladder_bundle.candidate_predictions
                    )
                )
            ),
        )
        max_local_error = max(
            max_local_error,
            float(
                np.max(
                    np.abs(
                        ref_bundle.local_prediction
                        - ladder_bundle.local_prediction
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
                        - ladder.field.outcome_counts
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
                        - ladder.cache.features
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
                        - ladder.cache.probabilities
                    )
                )
            ),
        )
        max_distance_error = max(
            max_distance_error,
            float(
                np.max(
                    np.abs(
                        reference.cache.distances
                        - ladder.cache.distances
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
                        - ladder.cache.kernels
                    )
                )
            ),
        )
        underflow_mismatches += int(
            np.count_nonzero(
                (reference.cache.kernels == 0.0)
                != (ladder.cache.kernels == 0.0)
            )
        )

        if not np.array_equal(ref_dirty, ladder_dirty):
            dirty_disagreements += 1
        if not np.array_equal(
            reference.current_indices,
            ladder.current_indices,
        ):
            structural_disagreements += 1

    steps = len(trace)
    ref_kernel_us = (
        reference.cache.kernel_compute_ns
        / steps
        / 1.0e3
    )
    ladder_kernel_us = (
        ladder.cache.kernel_compute_ns
        / steps
        / 1.0e3
    )
    ref_refresh_us = reference_refresh_ns / steps / 1.0e3
    ladder_refresh_us = ladder_refresh_ns / steps / 1.0e3
    ref_step_us = reference_step_ns / steps / 1.0e3
    ladder_step_us = ladder_step_ns / steps / 1.0e3

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
        "max_distance_error": max_distance_error,
        "max_kernel_error": max_kernel_error,
        "underflow_mismatches": underflow_mismatches,
        "dirty_disagreements": dirty_disagreements,
        "structural_disagreements": structural_disagreements,
        "reference_kernel_us_per_step": ref_kernel_us,
        "ladder_kernel_us_per_step": ladder_kernel_us,
        "kernel_speedup": (
            ref_kernel_us
            / max(ladder_kernel_us, 1.0e-15)
        ),
        "reference_refresh_us_per_step": ref_refresh_us,
        "ladder_refresh_us_per_step": ladder_refresh_us,
        "refresh_speedup": (
            ref_refresh_us
            / max(ladder_refresh_us, 1.0e-15)
        ),
        "reference_fused_us_per_step": ref_step_us,
        "ladder_fused_us_per_step": ladder_step_us,
        "fused_speedup": (
            ref_step_us
            / max(ladder_step_us, 1.0e-15)
        ),
    }


def _median(rows, key):
    return float(np.median([row[key] for row in rows]))


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
            "kernel_speedup": _median(
                selected,
                "kernel_speedup",
            ),
            "refresh_speedup": _median(
                selected,
                "refresh_speedup",
            ),
            "fused_speedup": _median(
                selected,
                "fused_speedup",
            ),
            "reference_kernel_us_per_step": _median(
                selected,
                "reference_kernel_us_per_step",
            ),
            "ladder_kernel_us_per_step": _median(
                selected,
                "ladder_kernel_us_per_step",
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
        "underflow_mismatches": sum(
            row["underflow_mismatches"]
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
            "kernel-ladder-fast-path-v1.18-R.jsonl"
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
