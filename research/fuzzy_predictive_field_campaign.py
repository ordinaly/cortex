"""Campaign for Cortex v1.6-R fuzzy predictive field."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dual_geometry_tensor import (
    fit_tensor_bridge,
    rarity_weights,
    relation_surprisal,
)
from fuzzy_predictive_field import FuzzyPredictiveField


FAMILIES = 4
INSTANCES = 3
SOURCE_COUNT = FAMILIES * INSTANCES
TARGET_COUNT = FAMILIES * INSTANCES
ENTITIES = SOURCE_COUNT + TARGET_COUNT
EMBEDDING_DIM = 24
OUTCOMES = 8
RELATIONS = 6


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def make_anchors(
    seed: int,
    *,
    individual_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    q, _r = np.linalg.qr(rng.normal(size=(EMBEDDING_DIM, 8)))
    family_basis = q[:, :8].T

    anchors = []
    families = []
    sides = []
    for side in (0, 1):
        for family in range(FAMILIES):
            base = family_basis[side * FAMILIES + family]
            for _instance in range(INSTANCES):
                residual = rng.normal(size=EMBEDDING_DIM)
                residual -= family_basis.T @ (family_basis @ residual)
                residual = _normalize(residual)
                center = _normalize(base + individual_scale * residual)
                anchors.append(center)
                families.append(family)
                sides.append(side)

    return (
        np.asarray(anchors),
        np.asarray(families),
        np.asarray(sides),
    )


def noisy_observation(
    anchor: np.ndarray,
    rng: np.random.Generator,
    *,
    noise: float,
) -> np.ndarray:
    return _normalize(
        anchor + rng.normal(0.0, noise, size=anchor.shape)
    )


def outcome_distribution(side: int, family: int) -> np.ndarray:
    q = np.full(OUTCOMES, 0.02, dtype=float)
    q[side * FAMILIES + family] = 0.78
    return q / q.sum()


def relation_distribution(source_family: int, target_family: int) -> np.ndarray:
    p = np.full(RELATIONS, 0.01, dtype=float)
    p[0] = 0.64
    p[1] = 0.18
    p[2 + ((source_family + target_family) % FAMILIES)] = 0.12
    return p / p.sum()


def heldout_pairs() -> set[tuple[int, int]]:
    return {
        (source_family * INSTANCES, SOURCE_COUNT + target_family * INSTANCES)
        for source_family in range(FAMILIES)
        for target_family in range(FAMILIES)
    }


def semantic_features(counts: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    probs = (counts + alpha) / (
        counts.sum(axis=1, keepdims=True) + alpha * OUTCOMES
    )
    return np.sqrt(probs)


def fit_bridge_from_counts(
    semantics: np.ndarray,
    pair_counts: np.ndarray,
):
    samples = []
    totals = pair_counts.sum(axis=2)
    for source in range(ENTITIES):
        for target in range(ENTITIES):
            total = float(totals[source, target])
            if total <= 1.0e-8:
                continue
            samples.append(
                (
                    source,
                    target,
                    (pair_counts[source, target] / total).tolist(),
                )
            )
    return fit_tensor_bridge(semantics, samples, ridge=1.0e-3)


def predictive_kernel(semantics: np.ndarray, resolution: float) -> np.ndarray:
    diff = semantics[:, None, :] - semantics[None, :, :]
    distances = np.linalg.norm(diff, axis=2) / np.sqrt(2.0)
    return np.exp(-0.5 * (distances / resolution) ** 2)


def effective_rank(kernel: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(0.5 * (kernel + kernel.T))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    p = eigenvalues / eigenvalues.sum()
    p = p[p > 1.0e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def run_case(
    seed: int,
    *,
    perception_noise: float = 0.16,
    individual_scale: float = 0.08,
    probes_per_entity: int = 15,
    interactions_per_pair: int = 3,
    semantic_resolution: float = 0.10,
) -> dict:
    rng = np.random.default_rng(seed)
    anchors, families, sides = make_anchors(
        seed + 991,
        individual_scale=individual_scale,
    )

    fuzzy = FuzzyPredictiveField(
        anchors,
        outcomes=OUTCOMES,
        relations=RELATIONS,
        perceptual_temperature=0.05,
        semantic_resolution=semantic_resolution,
    )

    hard_outcomes = np.zeros((ENTITIES, OUTCOMES), dtype=float)
    oracle_outcomes = np.zeros((ENTITIES, OUTCOMES), dtype=float)
    hard_pairs = np.zeros((ENTITIES, ENTITIES, RELATIONS), dtype=float)
    oracle_pairs = np.zeros((ENTITIES, ENTITIES, RELATIONS), dtype=float)
    global_relations = np.zeros(RELATIONS, dtype=float)

    exact_top1 = 0
    family_mass = []
    perceptual_samples = 0

    for entity in range(ENTITIES):
        for _ in range(probes_per_entity):
            observation = noisy_observation(
                anchors[entity],
                rng,
                noise=perception_noise,
            )
            possibility = fuzzy.possibility(observation)
            responsibility = possibility / possibility.sum()
            hard_identity = int(np.argmax(responsibility))

            exact_top1 += int(hard_identity == entity)
            family_mask = (
                (families == families[entity])
                & (sides == sides[entity])
            )
            family_mass.append(float(responsibility[family_mask].sum()))
            perceptual_samples += 1

            outcome = int(
                rng.choice(
                    OUTCOMES,
                    p=outcome_distribution(
                        int(sides[entity]),
                        int(families[entity]),
                    ),
                )
            )
            fuzzy.observe_outcome(observation, outcome)
            hard_outcomes[hard_identity, outcome] += 1.0
            oracle_outcomes[entity, outcome] += 1.0

    held = heldout_pairs()
    for source in range(SOURCE_COUNT):
        for target in range(SOURCE_COUNT, ENTITIES):
            if (source, target) in held:
                continue
            for _ in range(interactions_per_pair):
                source_observation = noisy_observation(
                    anchors[source],
                    rng,
                    noise=perception_noise,
                )
                target_observation = noisy_observation(
                    anchors[target],
                    rng,
                    noise=perception_noise,
                )
                source_weights = fuzzy.perceptual_weights(source_observation)
                target_weights = fuzzy.perceptual_weights(target_observation)
                hard_source = int(np.argmax(source_weights))
                hard_target = int(np.argmax(target_weights))

                relation = int(
                    rng.choice(
                        RELATIONS,
                        p=relation_distribution(
                            int(families[source]),
                            int(families[target]),
                        ),
                    )
                )
                fuzzy.observe_relation(
                    source_observation,
                    target_observation,
                    relation,
                )
                hard_pairs[hard_source, hard_target, relation] += 1.0
                oracle_pairs[source, target, relation] += 1.0
                global_relations[relation] += 1.0

    hard_semantics = semantic_features(hard_outcomes)
    oracle_semantics = semantic_features(oracle_outcomes)

    hard_bridge = fit_bridge_from_counts(hard_semantics, hard_pairs)
    oracle_bridge = fit_bridge_from_counts(oracle_semantics, oracle_pairs)
    oracle_kernel = predictive_kernel(
        oracle_semantics,
        semantic_resolution,
    )

    surprisal = relation_surprisal(global_relations + 1.0)
    eta_rare = float(np.mean(surprisal[2:]))
    rarity = rarity_weights(
        surprisal,
        eta_rare,
        bandwidth=0.45,
    )

    fuzzy_hits = []
    hard_hits = []
    oracle_hard_hits = []
    oracle_field_hits = []

    for source, target in sorted(held):
        source_observation = noisy_observation(
            anchors[source],
            rng,
            noise=perception_noise,
        )
        target_observation = noisy_observation(
            anchors[target],
            rng,
            noise=perception_noise,
        )

        fuzzy_scores = fuzzy.predict(
            source_observation,
            target_observation,
            eta=eta_rare,
        ).scores

        source_weights = fuzzy.perceptual_weights(source_observation)
        target_weights = fuzzy.perceptual_weights(target_observation)
        hard_source = int(np.argmax(source_weights))
        hard_target = int(np.argmax(target_weights))
        hard_scores = rarity * hard_bridge.predict(
            hard_semantics[hard_source],
            hard_semantics[hard_target],
        )

        oracle_hard_scores = rarity * oracle_bridge.predict(
            oracle_semantics[source],
            oracle_semantics[target],
        )

        source_field = oracle_kernel[source].copy()
        source_field /= source_field.sum()
        target_field = oracle_kernel[target].copy()
        target_field /= target_field.sum()
        oracle_field_scores = rarity * oracle_bridge.predict(
            source_field @ oracle_semantics,
            target_field @ oracle_semantics,
        )

        truth = 2 + (
            (int(families[source]) + int(families[target])) % FAMILIES
        )
        fuzzy_hits.append(int(int(np.argmax(fuzzy_scores)) == truth))
        hard_hits.append(int(int(np.argmax(hard_scores)) == truth))
        oracle_hard_hits.append(
            int(int(np.argmax(oracle_hard_scores)) == truth)
        )
        oracle_field_hits.append(
            int(int(np.argmax(oracle_field_scores)) == truth)
        )

    return {
        "protocol": "fuzzy-predictive-field-v1",
        "seed": seed,
        "perception_noise": perception_noise,
        "individual_scale": individual_scale,
        "probes_per_entity": probes_per_entity,
        "interactions_per_pair": interactions_per_pair,
        "semantic_resolution": semantic_resolution,
        "physical_anchors": ENTITIES,
        "exact_identity_top1": exact_top1 / perceptual_samples,
        "correct_family_possibility_mass": float(np.mean(family_mass)),
        "fuzzy_future_hit_at_1": float(np.mean(fuzzy_hits)),
        "hard_projection_future_hit_at_1": float(np.mean(hard_hits)),
        "oracle_hard_future_hit_at_1": float(np.mean(oracle_hard_hits)),
        "oracle_field_future_hit_at_1": float(np.mean(oracle_field_hits)),
        "predictive_effective_rank": fuzzy.predictive_complexity(),
    }


def campaign(seeds: int, output: Path) -> None:
    rows = []

    for seed in range(seeds):
        rows.append(run_case(seed))

    for noise in (0.08, 0.12, 0.20, 0.25):
        for seed in range(max(4, seeds // 2)):
            rows.append(
                run_case(
                    10000 + seed,
                    perception_noise=noise,
                )
            )

    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    base = [
        row
        for row in rows
        if row["seed"] < seeds
        and row["perception_noise"] == 0.16
    ]
    print(
        json.dumps(
            {
                "suite": "base",
                "seeds": len(base),
                "exact_identity_top1": float(
                    np.mean([row["exact_identity_top1"] for row in base])
                ),
                "correct_family_possibility_mass": float(
                    np.mean(
                        [
                            row["correct_family_possibility_mass"]
                            for row in base
                        ]
                    )
                ),
                "fuzzy_future_hit_at_1": float(
                    np.mean(
                        [row["fuzzy_future_hit_at_1"] for row in base]
                    )
                ),
                "hard_projection_future_hit_at_1": float(
                    np.mean(
                        [
                            row["hard_projection_future_hit_at_1"]
                            for row in base
                        ]
                    )
                ),
                "oracle_hard_future_hit_at_1": float(
                    np.mean(
                        [row["oracle_hard_future_hit_at_1"] for row in base]
                    )
                ),
                "oracle_field_future_hit_at_1": float(
                    np.mean(
                        [row["oracle_field_future_hit_at_1"] for row in base]
                    )
                ),
                "predictive_effective_rank": float(
                    np.mean(
                        [row["predictive_effective_rank"] for row in base]
                    )
                ),
            },
            sort_keys=True,
        )
    )

    for noise in (0.08, 0.12, 0.20, 0.25):
        subset = [
            row
            for row in rows
            if row["seed"] >= 10000
            and row["perception_noise"] == noise
        ]
        print(
            json.dumps(
                {
                    "suite": "noise-sweep",
                    "perception_noise": noise,
                    "seeds": len(subset),
                    "exact_identity_top1": float(
                        np.mean(
                            [row["exact_identity_top1"] for row in subset]
                        )
                    ),
                    "family_mass": float(
                        np.mean(
                            [
                                row["correct_family_possibility_mass"]
                                for row in subset
                            ]
                        )
                    ),
                    "fuzzy_hit1": float(
                        np.mean(
                            [
                                row["fuzzy_future_hit_at_1"]
                                for row in subset
                            ]
                        )
                    ),
                    "hard_hit1": float(
                        np.mean(
                            [
                                row["hard_projection_future_hit_at_1"]
                                for row in subset
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
        default=Path("fuzzy-predictive-field-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.seeds, args.output)


if __name__ == "__main__":
    main()
