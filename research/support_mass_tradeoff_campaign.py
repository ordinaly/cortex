"""Cortex v1.12-R retained-mass support tradeoff campaign."""
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
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
)


RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)
ANCHOR_CASES = (72, 144, 288)
RETAINED_MASSES = (0.99, 0.995, 0.999)
OUTCOMES = 3


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


def _controller(
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


def run_controller(
    anchors: np.ndarray,
    trace: list[tuple[np.ndarray, int, str]],
    *,
    retained_mass: float,
) -> dict:
    controller = _controller(
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

        local_predictions.append(
            bundle.local_prediction.copy()
        )
        candidate_predictions.append(
            bundle.candidate_predictions.copy()
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
        "local_predictions": np.asarray(
            local_predictions,
            dtype=float,
        ),
        "candidate_predictions": np.asarray(
            candidate_predictions,
            dtype=float,
        ),
        "resolutions": resolutions,
        "losses": np.asarray(losses, dtype=float),
        "phases": np.asarray(phases),
        "support": controller.support_diagnostics(),
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


def run_case(
    seed: int,
    anchors_count: int,
    retained_mass: float,
) -> dict:
    anchors = make_anchors(
        seed + 1301 + anchors_count,
        anchors=anchors_count,
        individual_scale=0.35,
    )
    trace = make_trace(
        seed + 4001,
        anchors,
    )

    baseline = run_controller(
        anchors,
        trace,
        retained_mass=1.0,
    )
    compressed = run_controller(
        anchors,
        trace,
        retained_mass=retained_mass,
    )

    local_tv = 0.5 * np.sum(
        np.abs(
            baseline["local_predictions"]
            - compressed["local_predictions"]
        ),
        axis=1,
    )
    candidate_tv = 0.5 * np.sum(
        np.abs(
            baseline["candidate_predictions"]
            - compressed["candidate_predictions"]
        ),
        axis=2,
    )
    divergent = (
        baseline["phases"] == "divergent"
    )
    reconverged_indexes = np.flatnonzero(
        compressed["phases"] == "reconverged"
    )
    reconverged_tail = reconverged_indexes[
        len(reconverged_indexes) // 2 :
    ]
    coarsest = max(RESOLUTIONS)
    recoarsened_fraction = float(
        np.mean(
            [
                all(
                    np.isclose(value, coarsest)
                    for value
                    in compressed["resolutions"][index]
                )
                for index in reconverged_tail
            ]
        )
    )

    return {
        "protocol": "support-mass-tradeoff-v1",
        "seed": seed,
        "anchors": anchors_count,
        "retained_mass": retained_mass,
        "steps": len(trace),
        "baseline_us_per_step": (
            1.0e6
            * baseline["seconds"]
            / len(trace)
        ),
        "compressed_us_per_step": (
            1.0e6
            * compressed["seconds"]
            / len(trace)
        ),
        "speedup_vs_full": (
            baseline["seconds"]
            / compressed["seconds"]
        ),
        "mean_local_tv": float(
            np.mean(local_tv)
        ),
        "max_local_tv": float(
            np.max(local_tv)
        ),
        "mean_candidate_tv": float(
            np.mean(candidate_tv)
        ),
        "max_candidate_tv": float(
            np.max(candidate_tv)
        ),
        "resolution_disagreement": (
            _resolution_disagreement(
                baseline["resolutions"],
                compressed["resolutions"],
            )
        ),
        "baseline_divergent_nll": float(
            np.mean(
                baseline["losses"][divergent]
            )
        ),
        "compressed_divergent_nll": float(
            np.mean(
                compressed["losses"][divergent]
            )
        ),
        "recoarsened_fraction": (
            recoarsened_fraction
        ),
        **compressed["support"],
    }


def _admissible(row: dict) -> bool:
    return (
        row["mean_local_tv"] <= 0.01
        and row["max_local_tv"] <= 0.05
        and row["mean_candidate_tv"] <= 0.01
        and row["max_candidate_tv"] <= 0.05
        and row["resolution_disagreement"] <= 0.05
        and row["compressed_divergent_nll"]
        <= row["baseline_divergent_nll"] + 0.02
        and row["recoarsened_fraction"] >= 0.95
        and row["max_dropped_mass"]
        <= 1.0 - row["retained_mass"] + 1.0e-12
    )


def campaign(seeds: int, output: Path) -> None:
    rows = [
        run_case(seed, anchors, retained_mass)
        for anchors in ANCHOR_CASES
        for retained_mass in RETAINED_MASSES
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
        "protocol": "support-mass-tradeoff-v1",
        "seeds": seeds,
        "anchor_cases": list(ANCHOR_CASES),
        "retained_masses": list(RETAINED_MASSES),
        "cases": {},
    }

    for anchors in ANCHOR_CASES:
        anchor_summary = {}
        for retained_mass in RETAINED_MASSES:
            selected = [
                row
                for row in rows
                if row["anchors"] == anchors
                and np.isclose(
                    row["retained_mass"],
                    retained_mass,
                )
            ]

            def median(key: str) -> float:
                return float(
                    np.median(
                        [row[key] for row in selected]
                    )
                )

            admissible_fraction = float(
                np.mean(
                    [
                        _admissible(row)
                        for row in selected
                    ]
                )
            )
            anchor_summary[str(retained_mass)] = {
                key: median(key)
                for key in (
                    "baseline_us_per_step",
                    "compressed_us_per_step",
                    "speedup_vs_full",
                    "mean_local_tv",
                    "max_local_tv",
                    "mean_candidate_tv",
                    "max_candidate_tv",
                    "resolution_disagreement",
                    "baseline_divergent_nll",
                    "compressed_divergent_nll",
                    "recoarsened_fraction",
                    "mean_support_fraction",
                    "max_support_fraction",
                    "mean_dropped_mass",
                    "max_dropped_mass",
                )
            }
            anchor_summary[str(retained_mass)][
                "admissible_seed_fraction"
            ] = admissible_fraction

        summary["cases"][str(anchors)] = (
            anchor_summary
        )

    best = None
    candidates = []
    for retained_mass in RETAINED_MASSES:
        selected = [
            row
            for row in rows
            if row["anchors"] == 288
            and np.isclose(
                row["retained_mass"],
                retained_mass,
            )
        ]
        if selected and all(
            _admissible(row)
            for row in selected
        ):
            candidates.append(
                (
                    float(
                        np.median(
                            [
                                row["compressed_us_per_step"]
                                for row in selected
                            ]
                        )
                    ),
                    retained_mass,
                )
            )
    if candidates:
        candidates.sort()
        best = candidates[0][1]

    summary[
        "fastest_fully_admissible_mass_at_288"
    ] = best

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
            "support-mass-tradeoff-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
