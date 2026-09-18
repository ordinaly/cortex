"""Hardness sweep for Cortex v1.3-R dual geometry.

Tests whether the perfect base ablation survives noisier semantic features and
reduced observed-pair coverage. This is a new regime, not extra repetitions of
the easy case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dual_geometry_ablation import (
    NEAR,
    make_world,
)
from dual_geometry_tensor import (
    fit_tensor_bridge,
    rarity_weights,
)


def evaluate(
    seed: int,
    *,
    semantic_noise: float,
    train_fraction: float,
    rate_noise_scale: float = 1.0,
) -> dict:
    semantics, samples, heldout, _rates, surprisal = make_world(
        seed,
        semantic_noise=semantic_noise,
        rate_noise_scale=rate_noise_scale,
    )

    rng = np.random.default_rng(
        seed
        + int(round(semantic_noise * 10000)) * 101
        + int(round(train_fraction * 1000)) * 1009
    )
    keep = max(1, int(round(len(samples) * train_fraction)))
    indices = rng.permutation(len(samples))[:keep]
    train = [samples[int(index)] for index in indices]

    bridge = fit_tensor_bridge(semantics, train, ridge=1.0e-3)
    eta_rare = float(surprisal[2])
    eta_common = float(surprisal[NEAR])
    fine_w = rarity_weights(surprisal, eta_rare, bandwidth=0.35)
    coarse_w = rarity_weights(surprisal, eta_common, bandwidth=0.35)

    rare_hit = 0
    coarse_hit = 0
    margins = []
    for source, target, truth in heldout:
        rates = bridge.predict(semantics[source], semantics[target])
        fine = fine_w * rates
        coarse = coarse_w * rates
        rare_hit += int(int(np.argmax(fine)) == truth)
        coarse_hit += int(int(np.argmax(coarse)) == NEAR)

        rare = fine[2:]
        truth_index = truth - 2
        wrong = np.delete(rare, truth_index)
        margins.append(float(fine[truth] - np.max(wrong)))

    return {
        "protocol": "dual-geometry-hardness-v1",
        "seed": seed,
        "semantic_noise": semantic_noise,
        "train_fraction": train_fraction,
        "rate_noise_scale": rate_noise_scale,
        "train_pairs": len(train),
        "heldout_pairs": len(heldout),
        "fine_rare_hit_at_1": rare_hit / len(heldout),
        "coarse_near_hit_at_1": coarse_hit / len(heldout),
        "mean_rare_margin": float(np.mean(margins)),
    }


def campaign(seeds: int, output: Path) -> None:
    rows = []
    for semantic_noise in (0.025, 0.10, 0.20, 0.35):
        for train_fraction in (1.0, 0.50, 0.25, 0.125):
            for seed in range(seeds):
                rows.append(
                    evaluate(
                        seed,
                        semantic_noise=semantic_noise,
                        train_fraction=train_fraction,
                    )
                )

    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    for semantic_noise in (0.025, 0.10, 0.20, 0.35):
        for train_fraction in (1.0, 0.50, 0.25, 0.125):
            subset = [
                row
                for row in rows
                if row["semantic_noise"] == semantic_noise
                and row["train_fraction"] == train_fraction
            ]
            print(
                json.dumps(
                    {
                        "semantic_noise": semantic_noise,
                        "train_fraction": train_fraction,
                        "seeds": seeds,
                        "fine_rare_hit_at_1": float(
                            np.mean(
                                [row["fine_rare_hit_at_1"] for row in subset]
                            )
                        ),
                        "coarse_near_hit_at_1": float(
                            np.mean(
                                [row["coarse_near_hit_at_1"] for row in subset]
                            )
                        ),
                        "mean_rare_margin": float(
                            np.mean([row["mean_rare_margin"] for row in subset])
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
        default=Path("dual-geometry-hardness-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
