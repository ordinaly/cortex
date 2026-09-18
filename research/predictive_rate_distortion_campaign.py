"""Cortex v1.7-R adaptive predictive rate-distortion campaign.

Three perceptually ambiguous anchors pass through:
1. an equivalent-future phase;
2. a divergent-future phase;
3. a reconverged-future phase.

The controller must coarsen, sharpen, then coarsen again using only prequential
predictive loss and current field complexity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fuzzy_predictive_field import FuzzyPredictiveField
from predictive_rate_distortion import AdaptiveResolutionController


ANCHORS = 3
DIM = 12
OUTCOMES = 3
RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def make_anchors(
    seed: int,
    *,
    individual_scale: float = 0.25,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _normalize(rng.normal(size=DIM))
    anchors = []
    for _ in range(ANCHORS):
        residual = rng.normal(size=DIM)
        residual -= base * float(np.dot(base, residual))
        anchors.append(
            _normalize(base + individual_scale * _normalize(residual))
        )
    return np.asarray(anchors)


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
    return -float(np.log(max(float(prediction[outcome]), 1.0e-9)))


def _phase_summary(records: list[dict]) -> dict:
    if not records:
        return {}
    tail_start = len(records) // 2
    tail = records[tail_start:]
    resolutions = np.asarray([row["resolution"] for row in tail])
    complexities = np.asarray([row["complexity"] for row in tail])
    return {
        "mean_resolution_tail": float(np.mean(resolutions)),
        "coarsest_fraction_tail": float(np.mean(resolutions == max(RESOLUTIONS))),
        "finest_two_fraction_tail": float(
            np.mean(resolutions <= RESOLUTIONS[1])
        ),
        "mean_complexity_tail": float(np.mean(complexities)),
        "adaptive_nll": float(np.mean([row["adaptive_nll"] for row in records])),
        "fine_nll": float(np.mean([row["fine_nll"] for row in records])),
        "coarse_nll": float(np.mean([row["coarse_nll"] for row in records])),
        "switches": int(
            np.sum(
                np.diff(
                    np.asarray([row["resolution"] for row in records])
                )
                != 0
            )
        ),
    }


def run_seed(
    seed: int,
    *,
    perception_noise: float = 0.25,
    equivalent_cycles: int = 500,
    divergent_cycles: int = 800,
    reconverged_cycles: int = 1000,
) -> dict:
    rng = np.random.default_rng(seed)
    anchors = make_anchors(seed + 451)

    field = FuzzyPredictiveField(
        anchors,
        outcomes=OUTCOMES,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
        evidence_decay=0.998,
    )
    controller = AdaptiveResolutionController(
        field,
        resolutions=RESOLUTIONS,
        epsilon=0.04,
        loss_decay=0.995,
        warmup_observations=60,
        minimum_relative_mass=0.10,
    )

    stable_future = np.asarray([0.95, 0.04, 0.01])
    divergent_future = np.asarray([0.01, 0.04, 0.95])

    phases: dict[str, list[dict]] = {
        "equivalent": [],
        "divergent": [],
        "reconverged": [],
    }

    def run_phase(name: str, cycles: int, divergent: bool) -> None:
        for _cycle in range(cycles):
            for identity in rng.permutation(ANCHORS):
                observation = observe(
                    anchors[identity],
                    rng,
                    noise=perception_noise,
                )

                adaptive_prediction, decision = controller.predict_outcome(
                    observation
                )
                fine_prediction = field.predict_outcome(
                    observation,
                    resolution=min(RESOLUTIONS),
                )
                coarse_prediction = field.predict_outcome(
                    observation,
                    resolution=max(RESOLUTIONS),
                )

                distribution = (
                    divergent_future
                    if divergent and identity == 0
                    else stable_future
                )
                outcome = int(rng.choice(OUTCOMES, p=distribution))

                row = {
                    "resolution": decision.resolution,
                    "complexity": decision.complexity,
                    "distortion": decision.distortion,
                    "best_distortion": decision.best_distortion,
                    "adaptive_nll": _nll(adaptive_prediction, outcome),
                    "fine_nll": _nll(fine_prediction, outcome),
                    "coarse_nll": _nll(coarse_prediction, outcome),
                    "identity": int(identity),
                }
                phases[name].append(row)
                controller.observe_outcome(observation, outcome)

    run_phase("equivalent", equivalent_cycles, divergent=False)
    run_phase("divergent", divergent_cycles, divergent=True)
    run_phase("reconverged", reconverged_cycles, divergent=False)

    summaries = {
        name: _phase_summary(records)
        for name, records in phases.items()
    }

    final_decision = controller.decision()
    final_distortions = controller.distortions().tolist()
    final_complexities = controller.complexities().tolist()

    return {
        "protocol": "predictive-rate-distortion-v1",
        "seed": seed,
        "perception_noise": perception_noise,
        "epsilon": controller.epsilon,
        "resolutions": list(RESOLUTIONS),
        "equivalent": summaries["equivalent"],
        "divergent": summaries["divergent"],
        "reconverged": summaries["reconverged"],
        "final_resolution": final_decision.resolution,
        "final_distortions": final_distortions,
        "final_complexities": final_complexities,
        "final_admissible": list(final_decision.admissible),
    }


def campaign(seeds: int, output: Path) -> None:
    rows = [run_seed(seed) for seed in range(seeds)]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    def mean(path: tuple[str, str]) -> float:
        return float(
            np.mean([row[path[0]][path[1]] for row in rows])
        )

    print(
        json.dumps(
            {
                "seeds": seeds,
                "equivalent_mean_resolution": mean(
                    ("equivalent", "mean_resolution_tail")
                ),
                "equivalent_coarsest_fraction": mean(
                    ("equivalent", "coarsest_fraction_tail")
                ),
                "divergent_mean_resolution": mean(
                    ("divergent", "mean_resolution_tail")
                ),
                "divergent_finest_two_fraction": mean(
                    ("divergent", "finest_two_fraction_tail")
                ),
                "reconverged_mean_resolution": mean(
                    ("reconverged", "mean_resolution_tail")
                ),
                "reconverged_coarsest_fraction": mean(
                    ("reconverged", "coarsest_fraction_tail")
                ),
                "equivalent_complexity": mean(
                    ("equivalent", "mean_complexity_tail")
                ),
                "divergent_complexity": mean(
                    ("divergent", "mean_complexity_tail")
                ),
                "reconverged_complexity": mean(
                    ("reconverged", "mean_complexity_tail")
                ),
                "adaptive_nll_divergent": mean(
                    ("divergent", "adaptive_nll")
                ),
                "fine_nll_divergent": mean(
                    ("divergent", "fine_nll")
                ),
                "coarse_nll_divergent": mean(
                    ("divergent", "coarse_nll")
                ),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictive-rate-distortion-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
