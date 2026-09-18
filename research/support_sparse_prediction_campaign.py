"""Cortex v1.11-R fused support-sparse prediction campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from sparse_cached_resolution import (
    SparseCachedLocalResolutionController,
)
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
)


RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)
ANCHOR_CASES = (36, 72, 144, 288)
OUTCOMES = 3
RETAINED_MASS = 0.999


def _field(anchors: np.ndarray) -> FuzzyPredictiveField:
    return FuzzyPredictiveField(
        anchors,
        outcomes=OUTCOMES,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
        evidence_decay=0.998,
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
    rows: list[tuple[np.ndarray, int, str]] = []

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
                rng.choice(
                    OUTCOMES,
                    p=distribution,
                )
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


def _legacy_controller(
    anchors: np.ndarray,
):
    return SparseCachedLocalResolutionController(
        _field(anchors.copy()),
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
        loss_change_tolerance=0.001,
        full_scan_interval=64,
        complexity_refresh_interval=32,
    )


def _fused_controller(
    anchors: np.ndarray,
    *,
    retained_mass: float,
):
    return FusedSupportSparseLocalResolutionController(
        _field(anchors.copy()),
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
        loss_change_tolerance=0.001,
        full_scan_interval=64,
        complexity_refresh_interval=32,
        retained_mass=retained_mass,
    )


def run_legacy(
    anchors: np.ndarray,
    trace: list[tuple[np.ndarray, int, str]],
) -> dict:
    controller = _legacy_controller(anchors)
    local_predictions = []
    candidate_predictions = []
    resolutions = []
    losses = []
    phases = []
    elapsed = 0.0

    for x, outcome, phase in trace:
        start = perf_counter()
        candidates = controller.candidate_predictions(x)
        prediction, decision = controller.predict_outcome(x)
        controller.observe_outcome(
            x,
            outcome,
            candidate_predictions=candidates,
        )
        elapsed += perf_counter() - start

        candidate_predictions.append(candidates.copy())
        local_predictions.append(prediction.copy())
        resolutions.append(decision.resolutions)
        losses.append(
            -float(
                np.log(
                    max(
                        float(prediction[outcome]),
                        1.0e-9,
                    )
                )
            )
        )
        phases.append(phase)

    return {
        "seconds": elapsed,
        "candidate_predictions": np.asarray(
            candidate_predictions,
            dtype=float,
        ),
        "local_predictions": np.asarray(
            local_predictions,
            dtype=float,
        ),
        "resolutions": resolutions,
        "losses": np.asarray(losses, dtype=float),
        "phases": np.asarray(phases),
    }


def run_fused(
    anchors: np.ndarray,
    trace: list[tuple[np.ndarray, int, str]],
    *,
    retained_mass: float,
) -> dict:
    controller = _fused_controller(
        anchors,
        retained_mass=retained_mass,
    )
    local_predictions = []
    candidate_predictions = []
    resolutions = []
    losses = []
    phases = []
    elapsed = 0.0

    for x, outcome, phase in trace:
        start = perf_counter()
        bundle = controller.prepare(x)
        controller.commit(bundle, outcome)
        elapsed += perf_counter() - start

        candidate_predictions.append(
            bundle.candidate_predictions.copy()
        )
        local_predictions.append(
            bundle.local_prediction.copy()
        )
        resolutions.append(
            bundle.decision.resolutions
        )
        losses.append(
            -float(
                np.log(
                    max(
                        float(
                            bundle.local_prediction[
                                outcome
                            ]
                        ),
                        1.0e-9,
                    )
                )
            )
        )
        phases.append(phase)

    return {
        "seconds": elapsed,
        "candidate_predictions": np.asarray(
            candidate_predictions,
            dtype=float,
        ),
        "local_predictions": np.asarray(
            local_predictions,
            dtype=float,
        ),
        "resolutions": resolutions,
        "losses": np.asarray(losses, dtype=float),
        "phases": np.asarray(phases),
        "support": controller.support_diagnostics(),
        "optimization": (
            controller.optimization_diagnostics()
        ),
    }


def _resolution_disagreement(
    left: list[tuple[float, ...]],
    right: list[tuple[float, ...]],
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
        seed + 1109 + anchors_count,
        anchors=anchors_count,
        individual_scale=0.35,
    )
    trace = make_trace(
        seed + 3001,
        anchors,
    )

    legacy = run_legacy(
        anchors,
        trace,
    )
    fused = run_fused(
        anchors,
        trace,
        retained_mass=1.0,
    )
    support = run_fused(
        anchors,
        trace,
        retained_mass=RETAINED_MASS,
    )

    fused_candidate_error = float(
        np.max(
            np.abs(
                legacy["candidate_predictions"]
                - fused["candidate_predictions"]
            )
        )
    )
    fused_local_error = float(
        np.max(
            np.abs(
                legacy["local_predictions"]
                - fused["local_predictions"]
            )
        )
    )
    fused_resolution_disagreement = (
        _resolution_disagreement(
            legacy["resolutions"],
            fused["resolutions"],
        )
    )

    local_tv = 0.5 * np.sum(
        np.abs(
            fused["local_predictions"]
            - support["local_predictions"]
        ),
        axis=1,
    )
    candidate_tv = 0.5 * np.sum(
        np.abs(
            fused["candidate_predictions"]
            - support["candidate_predictions"]
        ),
        axis=2,
    )
    support_resolution_disagreement = (
        _resolution_disagreement(
            fused["resolutions"],
            support["resolutions"],
        )
    )
    divergent = (
        fused["phases"] == "divergent"
    )
    reconverged_indexes = np.flatnonzero(
        support["phases"] == "reconverged"
    )
    reconverged_tail = reconverged_indexes[
        len(reconverged_indexes) // 2 :
    ]
    coarsest = max(RESOLUTIONS)
    support_recoarsened_fraction = float(
        np.mean(
            [
                all(
                    np.isclose(value, coarsest)
                    for value
                    in support["resolutions"][index]
                )
                for index in reconverged_tail
            ]
        )
    )

    return {
        "protocol": "support-sparse-prediction-v1",
        "seed": seed,
        "anchors": anchors_count,
        "steps": len(trace),
        "retained_mass_target": RETAINED_MASS,
        "legacy_us_per_step": (
            1.0e6
            * legacy["seconds"]
            / len(trace)
        ),
        "fused_us_per_step": (
            1.0e6
            * fused["seconds"]
            / len(trace)
        ),
        "support_us_per_step": (
            1.0e6
            * support["seconds"]
            / len(trace)
        ),
        "fused_speedup_vs_legacy": (
            legacy["seconds"]
            / fused["seconds"]
        ),
        "support_speedup_vs_fused": (
            fused["seconds"]
            / support["seconds"]
        ),
        "support_speedup_vs_legacy": (
            legacy["seconds"]
            / support["seconds"]
        ),
        "fused_max_candidate_error": (
            fused_candidate_error
        ),
        "fused_max_local_error": (
            fused_local_error
        ),
        "fused_resolution_disagreement": (
            fused_resolution_disagreement
        ),
        "support_mean_local_tv": float(
            np.mean(local_tv)
        ),
        "support_max_local_tv": float(
            np.max(local_tv)
        ),
        "support_mean_candidate_tv": float(
            np.mean(candidate_tv)
        ),
        "support_max_candidate_tv": float(
            np.max(candidate_tv)
        ),
        "support_resolution_disagreement": (
            support_resolution_disagreement
        ),
        "fused_divergent_nll": float(
            np.mean(fused["losses"][divergent])
        ),
        "support_divergent_nll": float(
            np.mean(support["losses"][divergent])
        ),
        "support_recoarsened_fraction": (
            support_recoarsened_fraction
        ),
        **{
            "support_" + key: value
            for key, value
            in support["support"].items()
        },
        **{
            "support_opt_" + key: value
            for key, value
            in support["optimization"].items()
        },
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
        "protocol": "support-sparse-prediction-v1",
        "seeds": seeds,
        "retained_mass_target": RETAINED_MASS,
        "cases": {},
    }

    for anchors in ANCHOR_CASES:
        selected = [
            row
            for row in rows
            if row["anchors"] == anchors
        ]

        def median(key: str):
            return float(
                np.median(
                    [row[key] for row in selected]
                )
            )

        summary["cases"][str(anchors)] = {
            key: median(key)
            for key in (
                "legacy_us_per_step",
                "fused_us_per_step",
                "support_us_per_step",
                "fused_speedup_vs_legacy",
                "support_speedup_vs_fused",
                "support_speedup_vs_legacy",
                "fused_max_candidate_error",
                "fused_max_local_error",
                "fused_resolution_disagreement",
                "support_mean_local_tv",
                "support_max_local_tv",
                "support_mean_candidate_tv",
                "support_max_candidate_tv",
                "support_resolution_disagreement",
                "fused_divergent_nll",
                "support_divergent_nll",
                "support_recoarsened_fraction",
                "support_mean_support_fraction",
                "support_max_support_fraction",
                "support_mean_dropped_mass",
                "support_max_dropped_mass",
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
            "support-sparse-prediction-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
