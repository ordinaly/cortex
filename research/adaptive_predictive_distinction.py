"""Adaptive predictive-distinction campaign for Cortex v1.6-R.

Several anchors begin predictively equivalent. One anchor later develops a
different future-outcome distribution. No symbolic split operation is issued;
the field must sharpen continuously as accumulated predictive evidence diverges.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fuzzy_predictive_field import FuzzyPredictiveField


ANCHORS = 3
DIM = 12
OUTCOMES = 3


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
        residual = _normalize(residual)
        anchors.append(_normalize(base + individual_scale * residual))
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


def run_seed(
    seed: int,
    *,
    perception_noise: float = 0.25,
    pre_cycles: int = 500,
    post_cycles: int = 800,
) -> dict:
    rng = np.random.default_rng(seed)
    anchors = make_anchors(seed + 817)

    field = FuzzyPredictiveField(
        anchors,
        outcomes=OUTCOMES,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
        evidence_decay=1.0,
    )

    old_future = np.asarray([0.95, 0.04, 0.01])
    changed_future = np.asarray([0.01, 0.04, 0.95])

    exact_identity = []

    # Phase A: all three physical anchors are predictively equivalent.
    for _ in range(pre_cycles):
        for identity in rng.permutation(ANCHORS):
            observation = observe(
                anchors[identity],
                rng,
                noise=perception_noise,
            )
            weights = field.perceptual_weights(observation)
            exact_identity.append(int(np.argmax(weights)) == identity)
            outcome = int(rng.choice(OUTCOMES, p=old_future))
            field.observe_outcome(observation, outcome)

    before_kernel = field.kernel()
    before_rank = field.predictive_complexity()
    before_prediction = field.predict_outcome(anchors[0])

    # Phase B: anchor 0 develops a different future.
    field.evidence_decay = 0.995
    checkpoints = []

    for step in range(post_cycles):
        for identity in rng.permutation(ANCHORS):
            observation = observe(
                anchors[identity],
                rng,
                noise=perception_noise,
            )
            distribution = (
                changed_future if identity == 0 else old_future
            )
            outcome = int(rng.choice(OUTCOMES, p=distribution))
            field.observe_outcome(observation, outcome)

        if step + 1 in (50, 100, 200, 400, 800):
            kernel = field.kernel()
            checkpoints.append(
                {
                    "cycles": step + 1,
                    "changed_to_stable_affinity": float(kernel[0, 1]),
                    "stable_to_stable_affinity": float(kernel[1, 2]),
                    "effective_rank": field.predictive_complexity(),
                    "changed_future_probability": float(
                        field.predict_outcome(anchors[0])[2]
                    ),
                    "stable_future_probability": float(
                        field.predict_outcome(anchors[1])[0]
                    ),
                }
            )

    after_kernel = field.kernel()
    after_rank = field.predictive_complexity()
    changed_prediction = field.predict_outcome(anchors[0])
    stable_prediction = field.predict_outcome(anchors[1])

    return {
        "protocol": "adaptive-predictive-distinction-v1",
        "seed": seed,
        "perception_noise": perception_noise,
        "exact_identity_top1": float(np.mean(exact_identity)),
        "before_changed_to_stable_affinity": float(before_kernel[0, 1]),
        "before_stable_to_stable_affinity": float(before_kernel[1, 2]),
        "after_changed_to_stable_affinity": float(after_kernel[0, 1]),
        "after_stable_to_stable_affinity": float(after_kernel[1, 2]),
        "before_effective_rank": before_rank,
        "after_effective_rank": after_rank,
        "before_changed_future_probability": float(before_prediction[2]),
        "after_changed_future_probability": float(changed_prediction[2]),
        "after_stable_future_probability": float(stable_prediction[0]),
        "checkpoints": checkpoints,
    }


def campaign(seeds: int, output: Path) -> None:
    rows = [run_seed(seed) for seed in range(seeds)]
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "seeds": seeds,
                "exact_identity_top1": float(
                    np.mean([row["exact_identity_top1"] for row in rows])
                ),
                "before_affinity": float(
                    np.mean(
                        [
                            row["before_changed_to_stable_affinity"]
                            for row in rows
                        ]
                    )
                ),
                "after_affinity": float(
                    np.mean(
                        [
                            row["after_changed_to_stable_affinity"]
                            for row in rows
                        ]
                    )
                ),
                "stable_affinity_after": float(
                    np.mean(
                        [
                            row["after_stable_to_stable_affinity"]
                            for row in rows
                        ]
                    )
                ),
                "before_effective_rank": float(
                    np.mean(
                        [row["before_effective_rank"] for row in rows]
                    )
                ),
                "after_effective_rank": float(
                    np.mean(
                        [row["after_effective_rank"] for row in rows]
                    )
                ),
                "changed_future_probability_before": float(
                    np.mean(
                        [
                            row["before_changed_future_probability"]
                            for row in rows
                        ]
                    )
                ),
                "changed_future_probability_after": float(
                    np.mean(
                        [
                            row["after_changed_future_probability"]
                            for row in rows
                        ]
                    )
                ),
                "stable_future_probability_after": float(
                    np.mean(
                        [
                            row["after_stable_future_probability"]
                            for row in rows
                        ]
                    )
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
        default=Path("adaptive-predictive-distinction-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
