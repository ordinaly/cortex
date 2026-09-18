"""Cortex v1.14-R normalized-anchor perceptual fast-path campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fuzzy_predictive_field import (
    FuzzyPredictiveField,
    perceptual_possibility,
)
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
)


RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)
ANCHOR_CASES = (72, 144, 288, 576)
OUTCOMES = 3


class LegacyPerceptualField(FuzzyPredictiveField):
    """Explicit pre-v1.14 perceptual path for differential timing."""

    def possibility(self, observation):
        return perceptual_possibility(
            observation,
            self.anchors,
            temperature=self.perceptual_temperature,
        )


def _field(
    anchors: np.ndarray,
    *,
    legacy: bool,
) -> FuzzyPredictiveField:
    cls = (
        LegacyPerceptualField
        if legacy
        else FuzzyPredictiveField
    )
    return cls(
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
    legacy: bool,
):
    return FusedSupportSparseLocalResolutionController(
        _field(
            anchors.copy(),
            legacy=legacy,
        ),
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
):
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


def run_controller(
    anchors: np.ndarray,
    trace,
    *,
    legacy: bool,
) -> dict:
    controller = _controller(
        anchors,
        legacy=legacy,
    )
    local_predictions = []
    candidate_predictions = []
    resolutions = []
    elapsed = 0.0

    for observation, outcome, _phase in trace:
        start = perf_counter()
        bundle = controller.prepare(observation)
        controller.commit(bundle, outcome)
        elapsed += perf_counter() - start

        local_predictions.append(
            bundle.local_prediction.copy()
        )
        candidate_predictions.append(
            bundle.candidate_predictions.copy()
        )
        resolutions.append(
            bundle.decision.resolutions
        )

    return {
        "seconds": elapsed,
        "local_predictions": np.asarray(
            local_predictions,
            dtype=float,
        ),
        "candidate_predictions": np.asarray(
            candidate_predictions,
            dtype=float,
        ),
        "resolutions": resolutions,
        "outcome_counts": (
            controller.field.outcome_counts.copy()
        ),
        "loss_numerator": (
            controller.tracker.loss_numerator.copy()
        ),
        "anchor_mass": (
            controller.tracker.anchor_mass.copy()
        ),
    }


def benchmark_perception(
    anchors: np.ndarray,
    observations: list[np.ndarray],
) -> dict:
    legacy = _field(
        anchors.copy(),
        legacy=True,
    )
    fast = _field(
        anchors.copy(),
        legacy=False,
    )

    start = perf_counter()
    legacy_checksum = 0.0
    for observation in observations:
        legacy_checksum += float(
            legacy.perceptual_weights(
                observation
            )[0]
        )
    legacy_seconds = perf_counter() - start

    start = perf_counter()
    fast_checksum = 0.0
    for observation in observations:
        fast_checksum += float(
            fast.perceptual_weights(
                observation
            )[0]
        )
    fast_seconds = perf_counter() - start

    return {
        "legacy_seconds": legacy_seconds,
        "fast_seconds": fast_seconds,
        "speedup": (
            legacy_seconds / fast_seconds
        ),
        "checksum_error": abs(
            legacy_checksum - fast_checksum
        ),
    }


def _resolution_disagreement(
    left,
    right,
) -> float:
    return float(
        np.mean(
            [
                a != b
                for a, b in zip(left, right)
            ]
        )
    )


def run_case(seed: int, anchors_count: int) -> dict:
    anchors = make_anchors(
        seed + 1901 + anchors_count,
        anchors=anchors_count,
        individual_scale=0.35,
    )
    trace = make_trace(
        seed + 6007,
        anchors,
    )

    legacy = run_controller(
        anchors,
        trace,
        legacy=True,
    )
    fast = run_controller(
        anchors,
        trace,
        legacy=False,
    )

    observations = [
        row[0]
        for row in trace[: min(128, len(trace))]
    ]
    perception = benchmark_perception(
        anchors,
        observations,
    )

    return {
        "protocol": "normalized-anchor-perception-v1",
        "seed": seed,
        "anchors": anchors_count,
        "steps": len(trace),
        "legacy_us_per_step": (
            1.0e6
            * legacy["seconds"]
            / len(trace)
        ),
        "fast_us_per_step": (
            1.0e6
            * fast["seconds"]
            / len(trace)
        ),
        "overall_speedup": (
            legacy["seconds"]
            / fast["seconds"]
        ),
        "perception_speedup": (
            perception["speedup"]
        ),
        "perception_checksum_error": (
            perception["checksum_error"]
        ),
        "max_candidate_error": float(
            np.max(
                np.abs(
                    legacy["candidate_predictions"]
                    - fast["candidate_predictions"]
                )
            )
        ),
        "max_local_error": float(
            np.max(
                np.abs(
                    legacy["local_predictions"]
                    - fast["local_predictions"]
                )
            )
        ),
        "resolution_disagreement": (
            _resolution_disagreement(
                legacy["resolutions"],
                fast["resolutions"],
            )
        ),
        "outcome_count_error": float(
            np.max(
                np.abs(
                    legacy["outcome_counts"]
                    - fast["outcome_counts"]
                )
            )
        ),
        "loss_numerator_error": float(
            np.max(
                np.abs(
                    legacy["loss_numerator"]
                    - fast["loss_numerator"]
                )
            )
        ),
        "anchor_mass_error": float(
            np.max(
                np.abs(
                    legacy["anchor_mass"]
                    - fast["anchor_mass"]
                )
            )
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
        "protocol": "normalized-anchor-perception-v1",
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

        summary["cases"][str(anchors)] = {
            key: median(key)
            for key in (
                "legacy_us_per_step",
                "fast_us_per_step",
                "overall_speedup",
                "perception_speedup",
                "perception_checksum_error",
                "max_candidate_error",
                "max_local_error",
                "resolution_disagreement",
                "outcome_count_error",
                "loss_numerator_error",
                "anchor_mass_error",
            )
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
            "normalized-anchor-perception-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
