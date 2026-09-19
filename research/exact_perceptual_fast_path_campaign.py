"""Cortex v1.14-R exact perceptual fast-path campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

import numpy as np

from fused_stage_attribution_campaign import (
    ANCHOR_CASES,
    RESOLUTIONS,
    make_trace,
    run_case as stage_attribution_case,
)
from fuzzy_predictive_field import (
    FuzzyPredictiveField,
    perceptual_possibility,
)
from hysteretic_local_resolution_campaign import make_anchors
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
)

OUTCOMES = 3


class LegacyFuzzyPredictiveField(FuzzyPredictiveField):
    """Reference path that re-normalizes stored anchors every observation."""

    def possibility(self, observation):
        return perceptual_possibility(
            observation,
            self.anchors,
            temperature=self.perceptual_temperature,
        )


def _field(field_type, anchors: np.ndarray):
    return field_type(
        anchors,
        outcomes=OUTCOMES,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
        evidence_decay=0.998,
    )


def _controller(field):
    return FusedSupportSparseLocalResolutionController(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
        loss_change_tolerance=0.001,
        full_scan_interval=64,
        complexity_refresh_interval=32,
        retained_mass=1.0,
    )


def _timed(fn: Callable):
    start = perf_counter_ns()
    value = fn()
    return value, perf_counter_ns() - start


def _run_controller_step(
    controller: FusedSupportSparseLocalResolutionController,
    observation: np.ndarray,
    outcome: int,
):
    controller._advance_state()
    bundle = controller.prepare(observation)
    controller.commit(bundle, outcome)
    return bundle


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

    legacy = _controller(
        _field(
            LegacyFuzzyPredictiveField,
            anchors.copy(),
        )
    )
    fast = _controller(
        _field(
            FuzzyPredictiveField,
            anchors.copy(),
        )
    )

    max_weight_error = 0.0
    max_candidate_error = 0.0
    max_local_error = 0.0
    max_outcome_count_error = 0.0
    structural_disagreements = 0

    legacy_perceptual_ns = 0
    fast_perceptual_ns = 0
    legacy_step_ns = 0
    fast_step_ns = 0

    for index, (observation, outcome, _phase) in enumerate(trace):
        if index % 2 == 0:
            legacy_weights, elapsed = _timed(
                lambda: legacy.field.perceptual_weights(
                    observation
                )
            )
            legacy_perceptual_ns += elapsed
            fast_weights, elapsed = _timed(
                lambda: fast.field.perceptual_weights(
                    observation
                )
            )
            fast_perceptual_ns += elapsed
        else:
            fast_weights, elapsed = _timed(
                lambda: fast.field.perceptual_weights(
                    observation
                )
            )
            fast_perceptual_ns += elapsed
            legacy_weights, elapsed = _timed(
                lambda: legacy.field.perceptual_weights(
                    observation
                )
            )
            legacy_perceptual_ns += elapsed

        max_weight_error = max(
            max_weight_error,
            float(
                np.max(
                    np.abs(
                        legacy_weights - fast_weights
                    )
                )
            ),
        )

        if index % 2 == 0:
            legacy_bundle, elapsed = _timed(
                lambda: _run_controller_step(
                    legacy,
                    observation,
                    outcome,
                )
            )
            legacy_step_ns += elapsed
            fast_bundle, elapsed = _timed(
                lambda: _run_controller_step(
                    fast,
                    observation,
                    outcome,
                )
            )
            fast_step_ns += elapsed
        else:
            fast_bundle, elapsed = _timed(
                lambda: _run_controller_step(
                    fast,
                    observation,
                    outcome,
                )
            )
            fast_step_ns += elapsed
            legacy_bundle, elapsed = _timed(
                lambda: _run_controller_step(
                    legacy,
                    observation,
                    outcome,
                )
            )
            legacy_step_ns += elapsed

        max_weight_error = max(
            max_weight_error,
            float(
                np.max(
                    np.abs(
                        legacy_bundle.full_weights
                        - fast_bundle.full_weights
                    )
                )
            ),
        )
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

        if not np.array_equal(
            legacy.current_indices,
            fast.current_indices,
        ):
            structural_disagreements += 1

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

    steps = len(trace)
    legacy_perceptual_us = (
        legacy_perceptual_ns / steps / 1.0e3
    )
    fast_perceptual_us = (
        fast_perceptual_ns / steps / 1.0e3
    )
    legacy_step_us = legacy_step_ns / steps / 1.0e3
    fast_step_us = fast_step_ns / steps / 1.0e3

    row = {
        "protocol": "exact-perceptual-fast-path-v1.14-R",
        "seed": seed,
        "anchors": anchors_count,
        "steps": steps,
        "max_weight_error": max_weight_error,
        "max_candidate_error": max_candidate_error,
        "max_local_error": max_local_error,
        "max_outcome_count_error": max_outcome_count_error,
        "structural_disagreements": structural_disagreements,
        "legacy_perceptual_us_per_step": (
            legacy_perceptual_us
        ),
        "fast_perceptual_us_per_step": (
            fast_perceptual_us
        ),
        "perceptual_speedup": (
            legacy_perceptual_us
            / max(fast_perceptual_us, 1.0e-15)
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
                "stage_perceptual_us_per_step": (
                    stage["perceptual_us_per_step"]
                ),
                "stage_perceptual_share": (
                    stage["perceptual_share"]
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
            }
        )

    return row


def _median(rows, key):
    return float(
        np.median(
            [row[key] for row in rows]
        )
    )


def campaign(
    seeds: int,
    output: Path,
) -> dict:
    rows = [
        run_pair_case(seed, anchors)
        for anchors in ANCHOR_CASES
        for seed in range(seeds)
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
            "perceptual_speedup": _median(
                selected,
                "perceptual_speedup",
            ),
            "fused_speedup": _median(
                selected,
                "fused_speedup",
            ),
            "stage_perceptual_share": _median(
                selected,
                "stage_perceptual_share",
            ),
            "stage_perceptual_us_per_step": _median(
                selected,
                "stage_perceptual_us_per_step",
            ),
            "stage_total_us_per_step": _median(
                selected,
                "stage_total_us_per_step",
            ),
            "stage_accounted_fraction": _median(
                selected,
                "stage_accounted_fraction",
            ),
        }

    summary = {
        "protocol": "exact-perceptual-fast-path-v1.14-R",
        "seeds": seeds,
        "anchor_cases": list(ANCHOR_CASES),
        "max_weight_error": max(
            row["max_weight_error"]
            for row in rows
        ),
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
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "exact-perceptual-fast-path-v1.14-R.jsonl"
        ),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
