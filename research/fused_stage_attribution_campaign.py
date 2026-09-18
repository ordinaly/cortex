"""Cortex v1.13-R fused prequential stage-attribution campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
)


RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)
ANCHOR_CASES = (72, 144, 288)
OUTCOMES = 3
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


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    fallback = totals[:, 0] <= 1.0e-15
    normalized = values / np.maximum(totals, 1.0e-15)
    if np.any(fallback):
        normalized[fallback] = 1.0 / values.shape[1]
    return normalized


def candidate_from_weights(
    controller: FusedSupportSparseLocalResolutionController,
    weights: np.ndarray,
) -> np.ndarray:
    cache = controller.cache
    activations = np.einsum(
        "a,rab->rb",
        weights,
        cache.kernels,
        optimize=True,
    )
    semantic_fields = _normalize_rows(activations)
    return _normalize_rows(
        semantic_fields @ cache.probabilities
    )


def local_from_weights(
    controller: FusedSupportSparseLocalResolutionController,
    weights: np.ndarray,
) -> np.ndarray:
    cache = controller.cache
    rows = np.arange(len(controller.current_indices))
    operator = cache.kernels[
        controller.current_indices,
        rows,
        :,
    ]
    activation = weights @ operator
    total = float(activation.sum())
    if total <= 1.0e-15:
        semantic = np.full(
            len(controller.field.anchors),
            1.0 / len(controller.field.anchors),
        )
    else:
        semantic = activation / total

    prediction = semantic @ cache.probabilities
    prediction_total = float(prediction.sum())
    if prediction_total <= 1.0e-15:
        return np.full(
            controller.field.outcomes,
            1.0 / controller.field.outcomes,
        )
    return prediction / prediction_total


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


def make_trace(
    seed: int,
    anchors: np.ndarray,
    *,
    equivalent_steps: int = 60,
    divergent_steps: int = 90,
    reconverged_steps: int = 120,
) -> list[tuple[np.ndarray, int, str]]:
    rng = np.random.default_rng(seed)
    stable = np.asarray([0.95, 0.04, 0.01])
    divergent = np.asarray([0.01, 0.04, 0.95])
    rows = []

    def add_phase(
        name: str,
        steps: int,
        *,
        is_divergent: bool,
    ) -> None:
        for _ in range(steps):
            identity = int(
                rng.integers(0, len(anchors))
            )
            x = observe(
                anchors[identity],
                rng,
                noise=0.20,
            )
            distribution = (
                divergent
                if is_divergent and identity == 0
                else stable
            )
            outcome = int(
                rng.choice(OUTCOMES, p=distribution)
            )
            rows.append((x, outcome, name))

    add_phase(
        "equivalent",
        equivalent_steps,
        is_divergent=False,
    )
    add_phase(
        "divergent",
        divergent_steps,
        is_divergent=True,
    )
    add_phase(
        "reconverged",
        reconverged_steps,
        is_divergent=False,
    )
    return rows


def _elapsed_ns(fn):
    start = perf_counter_ns()
    value = fn()
    return value, perf_counter_ns() - start


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
    controller = _controller(anchors)

    totals = {
        stage: 0
        for stage in STAGES
    }
    total_ns = 0
    prediction_checksum = np.zeros(
        OUTCOMES,
        dtype=float,
    )

    for observation, outcome, _phase in trace:
        step_start = perf_counter_ns()

        _, elapsed = _elapsed_ns(
            controller._advance_state
        )
        totals["hysteresis_scan"] += elapsed

        complexity, elapsed = _elapsed_ns(
            controller._complexity
        )
        totals["complexity"] += elapsed
        if not np.isfinite(complexity):
            raise ValueError("non-finite complexity")

        weights, elapsed = _elapsed_ns(
            lambda: controller.field.perceptual_weights(
                observation
            )
        )
        totals["perceptual"] += elapsed

        candidates, elapsed = _elapsed_ns(
            lambda: candidate_from_weights(
                controller,
                weights,
            )
        )
        totals["candidate_readout"] += elapsed

        local_prediction, elapsed = _elapsed_ns(
            lambda: local_from_weights(
                controller,
                weights,
            )
        )
        totals["local_readout"] += elapsed

        _, elapsed = _elapsed_ns(
            lambda: controller._update_tracker_from_weights(
                weights,
                outcome,
                candidates,
            )
        )
        totals["tracker_update"] += elapsed

        _, elapsed = _elapsed_ns(
            lambda: controller._update_field_from_weights(
                weights,
                outcome,
            )
        )
        totals["field_update"] += elapsed

        _, elapsed = _elapsed_ns(
            controller.cache.refresh
        )
        totals["cache_refresh"] += elapsed

        prediction_checksum += local_prediction
        total_ns += perf_counter_ns() - step_start

    accounted = sum(totals.values())
    remainder = max(0, total_ns - accounted)
    steps = len(trace)

    stage_us = {
        stage: (
            totals[stage] / steps / 1.0e3
        )
        for stage in STAGES
    }
    stage_shares = {
        stage: (
            totals[stage] / total_ns
            if total_ns > 0
            else 0.0
        )
        for stage in STAGES
    }

    return {
        "protocol": "fused-stage-attribution-v1",
        "seed": seed,
        "anchors": anchors_count,
        "steps": steps,
        "total_us_per_step": (
            total_ns / steps / 1.0e3
        ),
        "accounted_fraction": (
            accounted / total_ns
            if total_ns > 0
            else 0.0
        ),
        "remainder_us_per_step": (
            remainder / steps / 1.0e3
        ),
        "prediction_checksum": (
            prediction_checksum.tolist()
        ),
        **{
            f"{stage}_us_per_step": value
            for stage, value in stage_us.items()
        },
        **{
            f"{stage}_share": value
            for stage, value in stage_shares.items()
        },
        "cache_row_refresh_fraction": (
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


def campaign(seeds: int, output: Path) -> None:
    rows = [
        run_case(seed, anchors)
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

    summary = {
        "protocol": "fused-stage-attribution-v1",
        "seeds": seeds,
        "anchor_cases": list(ANCHOR_CASES),
        "cases": {},
    }

    for anchors in ANCHOR_CASES:
        selected = [
            row
            for row in rows
            if row["anchors"] == anchors
        ]

        def median(key: str) -> float:
            return float(
                np.median(
                    [row[key] for row in selected]
                )
            )

        stages = {
            stage: {
                "us_per_step": median(
                    f"{stage}_us_per_step"
                ),
                "share": median(
                    f"{stage}_share"
                ),
            }
            for stage in STAGES
        }
        dominant = max(
            STAGES,
            key=lambda stage: stages[stage]["share"],
        )

        summary["cases"][str(anchors)] = {
            "total_us_per_step": median(
                "total_us_per_step"
            ),
            "accounted_fraction": median(
                "accounted_fraction"
            ),
            "remainder_us_per_step": median(
                "remainder_us_per_step"
            ),
            "dominant_stage": dominant,
            "stages": stages,
            "cache_row_refresh_fraction": median(
                "cache_row_refresh_fraction"
            ),
            "anchor_scan_fraction": median(
                "anchor_scan_fraction"
            ),
            "svd_fraction": median(
                "svd_fraction"
            ),
        }

    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "fused-stage-attribution-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
