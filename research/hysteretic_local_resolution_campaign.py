"""Cortex v1.8-R hysteresis + local adaptive resolution campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution import (
    HystereticGlobalResolutionController,
    HystereticLocalResolutionController,
)
from predictive_rate_distortion import AdaptiveResolutionController


DIM = 12
OUTCOMES = 3
ANCHORS = 9
RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def make_anchors(
    seed: int,
    *,
    anchors: int = ANCHORS,
    individual_scale: float = 0.35,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _normalize(rng.normal(size=DIM))
    rows = []
    for _ in range(anchors):
        residual = rng.normal(size=DIM)
        residual -= base * float(np.dot(base, residual))
        rows.append(
            _normalize(
                base
                + individual_scale * _normalize(residual)
            )
        )
    return np.asarray(rows)


def observe(
    anchor: np.ndarray,
    rng: np.random.Generator,
    *,
    noise: float,
) -> np.ndarray:
    return _normalize(
        anchor + rng.normal(0.0, noise, size=anchor.shape)
    )


def _nll(prediction: np.ndarray, outcome: int) -> float:
    return -float(
        np.log(max(float(prediction[outcome]), 1.0e-9))
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


def _global_summary(records: list[dict]) -> dict:
    tail = records[len(records) // 2 :]
    resolutions = np.asarray(
        [row["resolution"] for row in tail],
        dtype=float,
    )
    trace = np.asarray(
        [row["resolution"] for row in records],
        dtype=float,
    )
    return {
        "nll": float(
            np.mean([row["nll"] for row in records])
        ),
        "mean_resolution_tail": float(
            np.mean(resolutions)
        ),
        "coarsest_fraction_tail": float(
            np.mean(resolutions == max(RESOLUTIONS))
        ),
        "complexity_tail": float(
            np.mean([row["complexity"] for row in tail])
        ),
        "switches": int(
            np.sum(np.diff(trace) != 0)
        ),
    }


def _local_summary(records: list[dict]) -> dict:
    tail = records[len(records) // 2 :]
    resolutions = np.asarray(
        [row["resolutions"] for row in tail],
        dtype=float,
    )
    return {
        "nll": float(
            np.mean([row["nll"] for row in records])
        ),
        "mean_resolution_tail": float(
            np.mean(resolutions)
        ),
        "coarsest_anchor_fraction_tail": float(
            np.mean(resolutions == max(RESOLUTIONS))
        ),
        "refined_anchor_fraction_tail": float(
            np.mean(resolutions < max(RESOLUTIONS))
        ),
        "finest_two_anchor_fraction_tail": float(
            np.mean(resolutions <= RESOLUTIONS[1])
        ),
        "complexity_tail": float(
            np.mean([row["complexity"] for row in tail])
        ),
        "switches": int(records[-1]["switch_count"]),
    }


def run_seed(
    seed: int,
    *,
    perception_noise: float = 0.20,
    equivalent_cycles: int = 250,
    divergent_cycles: int = 400,
    reconverged_cycles: int = 500,
) -> dict:
    rng = np.random.default_rng(seed)
    anchors = make_anchors(seed + 451)

    reactive_field = _field(anchors)
    hysteretic_field = _field(anchors)
    local_field = _field(anchors)

    reactive = AdaptiveResolutionController(
        reactive_field,
        resolutions=RESOLUTIONS,
        epsilon=0.04,
        loss_decay=0.995,
        warmup_observations=60,
        minimum_relative_mass=0.10,
    )
    hysteretic = HystereticGlobalResolutionController(
        hysteretic_field,
        resolutions=RESOLUTIONS,
        epsilon=0.04,
        hysteresis_margin=0.01,
        coarsen_patience=12,
        loss_decay=0.995,
        warmup_observations=60,
        minimum_relative_mass=0.10,
    )
    local = HystereticLocalResolutionController(
        local_field,
        resolutions=RESOLUTIONS,
        epsilon=0.04,
        hysteresis_margin=0.01,
        coarsen_patience=12,
        loss_decay=0.995,
        warmup_observations=60,
        minimum_relative_mass=0.10,
    )

    stable_future = np.asarray([0.95, 0.04, 0.01])
    divergent_future = np.asarray([0.01, 0.04, 0.95])

    phases: dict[str, dict[str, list[dict]]] = {
        name: {
            "reactive": [],
            "hysteretic": [],
            "local": [],
            "fixed": [],
        }
        for name in (
            "equivalent",
            "divergent",
            "reconverged",
        )
    }

    def run_phase(
        name: str,
        cycles: int,
        *,
        divergent: bool,
    ) -> None:
        for _cycle in range(cycles):
            for identity in rng.permutation(ANCHORS):
                observation = observe(
                    anchors[identity],
                    rng,
                    noise=perception_noise,
                )

                reactive_decision = reactive.decision()
                reactive_candidates = (
                    reactive.candidate_predictions(observation)
                )
                reactive_index = reactive.resolutions.index(
                    reactive_decision.resolution
                )
                reactive_prediction = reactive_candidates[
                    reactive_index
                ]

                hysteretic_candidates = (
                    hysteretic.candidate_predictions(observation)
                )
                (
                    hysteretic_prediction,
                    hysteretic_decision,
                ) = hysteretic.predict_outcome(
                    observation,
                    candidate_predictions=hysteretic_candidates,
                )

                local_candidates = (
                    local.candidate_predictions(observation)
                )
                local_prediction, local_decision = (
                    local.predict_outcome(observation)
                )

                distribution = (
                    divergent_future
                    if divergent and identity == 0
                    else stable_future
                )
                outcome = int(
                    rng.choice(OUTCOMES, p=distribution)
                )

                phases[name]["reactive"].append(
                    {
                        "nll": _nll(
                            reactive_prediction,
                            outcome,
                        ),
                        "resolution": (
                            reactive_decision.resolution
                        ),
                        "complexity": (
                            reactive_decision.complexity
                        ),
                    }
                )
                phases[name]["hysteretic"].append(
                    {
                        "nll": _nll(
                            hysteretic_prediction,
                            outcome,
                        ),
                        "resolution": (
                            hysteretic_decision.resolution
                        ),
                        "complexity": (
                            hysteretic_decision.complexity
                        ),
                    }
                )
                phases[name]["local"].append(
                    {
                        "nll": _nll(
                            local_prediction,
                            outcome,
                        ),
                        "resolutions": list(
                            local_decision.resolutions
                        ),
                        "complexity": (
                            local_decision.complexity
                        ),
                        "switch_count": (
                            local_decision.switch_count
                        ),
                    }
                )
                phases[name]["fixed"].append(
                    {
                        "fine_nll": _nll(
                            reactive_candidates[0],
                            outcome,
                        ),
                        "coarse_nll": _nll(
                            reactive_candidates[-1],
                            outcome,
                        ),
                    }
                )

                reactive.observe_outcome(
                    observation,
                    outcome,
                    candidate_predictions=reactive_candidates,
                )
                hysteretic.observe_outcome(
                    observation,
                    outcome,
                    candidate_predictions=hysteretic_candidates,
                )
                local.observe_outcome(
                    observation,
                    outcome,
                    candidate_predictions=local_candidates,
                )

    run_phase(
        "equivalent",
        equivalent_cycles,
        divergent=False,
    )
    run_phase(
        "divergent",
        divergent_cycles,
        divergent=True,
    )
    run_phase(
        "reconverged",
        reconverged_cycles,
        divergent=False,
    )

    summaries = {}
    for name, records in phases.items():
        summaries[name] = {
            "reactive": _global_summary(
                records["reactive"]
            ),
            "hysteretic": _global_summary(
                records["hysteretic"]
            ),
            "local": _local_summary(
                records["local"]
            ),
            "fixed_fine_nll": float(
                np.mean(
                    [
                        row["fine_nll"]
                        for row in records["fixed"]
                    ]
                )
            ),
            "fixed_coarse_nll": float(
                np.mean(
                    [
                        row["coarse_nll"]
                        for row in records["fixed"]
                    ]
                )
            ),
        }

    return {
        "protocol": "hysteretic-local-resolution-v1",
        "seed": seed,
        "anchors": ANCHORS,
        "perception_noise": perception_noise,
        "resolutions": list(RESOLUTIONS),
        "epsilon": 0.04,
        "hysteresis_margin": 0.01,
        "coarsen_patience": 12,
        **summaries,
    }


def campaign(seeds: int, output: Path) -> None:
    rows = [run_seed(seed) for seed in range(seeds)]
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    def mean(
        phase: str,
        model: str,
        metric: str,
    ) -> float:
        return float(
            np.mean(
                [
                    row[phase][model][metric]
                    for row in rows
                ]
            )
        )

    summary = {
        "protocol": "hysteretic-local-resolution-v1",
        "seeds": seeds,
        "reactive_switches_divergent": mean(
            "divergent",
            "reactive",
            "switches",
        ),
        "hysteretic_switches_divergent": mean(
            "divergent",
            "hysteretic",
            "switches",
        ),
        "reactive_nll_divergent": mean(
            "divergent",
            "reactive",
            "nll",
        ),
        "hysteretic_nll_divergent": mean(
            "divergent",
            "hysteretic",
            "nll",
        ),
        "local_nll_divergent": mean(
            "divergent",
            "local",
            "nll",
        ),
        "hysteretic_complexity_divergent": mean(
            "divergent",
            "hysteretic",
            "complexity_tail",
        ),
        "local_complexity_divergent": mean(
            "divergent",
            "local",
            "complexity_tail",
        ),
        "local_refined_fraction_divergent": mean(
            "divergent",
            "local",
            "refined_anchor_fraction_tail",
        ),
        "local_coarsest_fraction_reconverged": mean(
            "reconverged",
            "local",
            "coarsest_anchor_fraction_tail",
        ),
    }
    print(json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "hysteretic-local-resolution-v1.jsonl"
        ),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
