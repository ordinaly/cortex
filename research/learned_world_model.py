"""Cortex v1.4-R learned semantic world-model campaign.

Hidden semantic families are used only by the generator and evaluation. The
predictor receives noisy identity embeddings, observed behavioral outcomes and
typed interaction events.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cortex import HybridCortexRuntime
from dual_geometry_tensor import (
    fit_tensor_bridge,
    rarity_weights,
    relation_surprisal,
)


FAMILIES = 4
INSTANCES = 3
SOURCE_COUNT = FAMILIES * INSTANCES
TARGET_COUNT = FAMILIES * INSTANCES
ENTITIES = SOURCE_COUNT + TARGET_COUNT
OUTCOMES = 8
RELATIONS = 6
NEAR = 0
CONTACT = 1
EMBEDDING_DIM = 32


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError("zero vector")
    return vector / norm


def perceptual_centers(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(EMBEDDING_DIM, ENTITIES))
    q, _r = np.linalg.qr(matrix)
    return q[:, :ENTITIES].T.copy()


def true_side_family(entity: int) -> tuple[int, int]:
    if entity < SOURCE_COUNT:
        return 0, entity // INSTANCES
    local = entity - SOURCE_COUNT
    return 1, local // INSTANCES


def probe_distribution(entity: int) -> np.ndarray:
    side, family = true_side_family(entity)
    primary = family if side == 0 else FAMILIES + family
    q = np.full(OUTCOMES, 0.03, dtype=float)
    q[side * FAMILIES : (side + 1) * FAMILIES] += 0.02
    q[primary] += 0.68
    q /= q.sum()
    return q


def interaction_distribution(source: int, target: int) -> np.ndarray:
    source_side, source_family = true_side_family(source)
    target_side, target_family = true_side_family(target)
    if source_side != 0 or target_side != 1:
        raise ValueError("interaction requires source-side -> target-side")
    truth = 2 + ((source_family + target_family) % FAMILIES)
    p = np.full(RELATIONS, 0.02, dtype=float)
    p[NEAR] = 0.64
    p[CONTACT] = 0.18
    p[truth] = 0.12
    p /= p.sum()
    return p


def heldout_true_pairs() -> list[tuple[int, int, int]]:
    result = []
    for sf in range(FAMILIES):
        source = sf * INSTANCES
        for tf in range(FAMILIES):
            target = SOURCE_COUNT + tf * INSTANCES
            truth = 2 + ((sf + tf) % FAMILIES)
            result.append((source, target, truth))
    return result


def noisy_observation(
    center: np.ndarray,
    rng: np.random.Generator,
    perception_noise: float,
) -> list[float]:
    return _normalize(
        center + rng.normal(0.0, perception_noise, size=center.shape)
    ).tolist()


class Experience:
    def __init__(self) -> None:
        self.semantic_counts: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(OUTCOMES, dtype=float)
        )
        self.perceptual_sum: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(EMBEDDING_DIM, dtype=float)
        )
        self.perceptual_n: Counter[int] = Counter()
        self.pair_counts: dict[tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(RELATIONS, dtype=float)
        )
        self.pair_exposures: Counter[tuple[int, int]] = Counter()
        self.source_relation_sum: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(RELATIONS, dtype=float)
        )
        self.source_exposures: Counter[int] = Counter()
        self.target_relation_sum: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(RELATIONS, dtype=float)
        )
        self.target_exposures: Counter[int] = Counter()
        self.global_relation_counts = np.zeros(RELATIONS, dtype=float)
        self.last_binding: dict[int, int] = {}
        self.identity_records: list[tuple[int, int]] = []

    def observe_binding(
        self,
        true_entity: int,
        binding: int,
        embedding: list[float],
    ) -> None:
        self.last_binding[true_entity] = binding
        self.identity_records.append((true_entity, binding))
        self.perceptual_sum[binding] += np.asarray(embedding, dtype=float)
        self.perceptual_n[binding] += 1

    def add_semantic_outcome(self, binding: int, outcome: int) -> None:
        self.semantic_counts[binding][outcome] += 1.0

    def add_interaction(self, source: int, target: int, relation: int) -> None:
        pair = (source, target)
        self.pair_counts[pair][relation] += 1.0
        self.pair_exposures[pair] += 1
        self.source_relation_sum[source][relation] += 1.0
        self.source_exposures[source] += 1
        self.target_relation_sum[target][relation] += 1.0
        self.target_exposures[target] += 1
        self.global_relation_counts[relation] += 1.0


def observe_single(
    runtime: HybridCortexRuntime,
    experience: Experience,
    true_entity: int,
    centers: np.ndarray,
    rng: np.random.Generator,
    perception_noise: float,
) -> int:
    embedding = noisy_observation(
        centers[true_entity], rng, perception_noise
    )
    read = runtime.step_embeddings([embedding])
    binding = int(read.bindings[0])
    experience.observe_binding(true_entity, binding, embedding)
    return binding


def observe_pair(
    runtime: HybridCortexRuntime,
    experience: Experience,
    source: int,
    target: int,
    centers: np.ndarray,
    rng: np.random.Generator,
    perception_noise: float,
) -> tuple[int, int]:
    items = [
        (
            source,
            noisy_observation(centers[source], rng, perception_noise),
        ),
        (
            target,
            noisy_observation(centers[target], rng, perception_noise),
        ),
    ]
    rng.shuffle(items)
    read = runtime.step_embeddings([embedding for _entity, embedding in items])
    by_true = {}
    for (true_entity, embedding), binding in zip(items, read.bindings):
        binding = int(binding)
        experience.observe_binding(true_entity, binding, embedding)
        by_true[true_entity] = binding
    return by_true[source], by_true[target]


def semantic_matrix(
    experience: Experience,
    *,
    alpha: float = 0.5,
) -> np.ndarray:
    max_id = max(
        [
            *experience.semantic_counts.keys(),
            *experience.last_binding.values(),
        ],
        default=-1,
    )
    matrix = np.zeros((max_id + 1, OUTCOMES), dtype=float)
    for entity in range(max_id + 1):
        counts = experience.semantic_counts[entity]
        probs = (counts + alpha) / (counts.sum() + alpha * OUTCOMES)
        matrix[entity] = np.sqrt(probs)
    return matrix


def identity_metrics(records: list[tuple[int, int]]) -> dict[str, float]:
    if not records:
        return {"purity": 0.0, "fragmentation": 0.0}
    by_binding: dict[int, Counter[int]] = defaultdict(Counter)
    by_truth: dict[int, set[int]] = defaultdict(set)
    for truth, binding in records:
        by_binding[binding][truth] += 1
        by_truth[truth].add(binding)
    correct = sum(max(counter.values()) for counter in by_binding.values())
    return {
        "purity": correct / len(records),
        "fragmentation": float(
            np.mean([len(bindings) for bindings in by_truth.values()])
        ),
    }


def semantic_separation(experience: Experience) -> dict[str, float]:
    matrix = semantic_matrix(experience)
    within = []
    between = []
    for a in range(ENTITIES):
        if a not in experience.last_binding:
            continue
        side_a, family_a = true_side_family(a)
        va = matrix[experience.last_binding[a]]
        for b in range(a + 1, ENTITIES):
            if b not in experience.last_binding:
                continue
            side_b, family_b = true_side_family(b)
            if side_a != side_b:
                continue
            vb = matrix[experience.last_binding[b]]
            distance = float(np.linalg.norm(va - vb) / np.sqrt(2.0))
            if family_a == family_b:
                within.append(distance)
            else:
                between.append(distance)
    within_mean = float(np.mean(within)) if within else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    return {
        "within_family": within_mean,
        "between_family": between_mean,
        "separation_ratio": (
            between_mean / max(within_mean, 1.0e-12)
            if between
            else 0.0
        ),
    }


def _tie_hit1(scores: np.ndarray, truth: int) -> float:
    truth_score = float(scores[truth])
    greater = int(np.sum(scores > truth_score + 1.0e-15))
    tied = int(np.sum(np.abs(scores - truth_score) <= 1.0e-15))
    best = greater + 1
    ranks = np.arange(best, best + tied)
    return float(np.mean(ranks <= 1))


def checkpoint_prediction(experience: Experience, *, rng_seed: int) -> dict:
    heldout = heldout_true_pairs()
    if any(
        source not in experience.last_binding or target not in experience.last_binding
        for source, target, _truth in heldout
    ):
        return {"available": False}

    semantics = semantic_matrix(experience)
    held_internal = {
        (
            experience.last_binding[source],
            experience.last_binding[target],
        )
        for source, target, _truth in heldout
    }

    samples = []
    for pair, counts in experience.pair_counts.items():
        exposures = experience.pair_exposures[pair]
        if exposures <= 0 or pair in held_internal:
            continue
        samples.append((pair[0], pair[1], (counts / exposures).tolist()))

    if len(samples) < 8:
        return {"available": False}

    bridge = fit_tensor_bridge(semantics, samples, ridge=1.0e-3)
    surprisal = relation_surprisal(experience.global_relation_counts + 1.0)
    eta_rare = float(np.mean(surprisal[2:]))
    weights = rarity_weights(surprisal, eta_rare, bandwidth=0.45)

    rng = np.random.default_rng(rng_seed)
    shuffled = semantics.copy()
    rng.shuffle(shuffled, axis=0)
    shuffled_bridge = fit_tensor_bridge(shuffled, samples, ridge=1.0e-3)

    methods = {
        "semantic-only": [],
        "interaction-marginal-only": [],
        "tensor-no-resolution": [],
        "full-learned-world-model": [],
        "shuffled-semantic-control": [],
    }

    internal_pair_collision = 0
    for true_source, true_target, truth in heldout:
        source = experience.last_binding[true_source]
        target = experience.last_binding[true_target]
        if experience.pair_exposures[(source, target)] > 0:
            internal_pair_collision += 1

        semantic_only = np.zeros(RELATIONS, dtype=float)
        semantic_only[2:] = 1.0

        source_rate = (
            experience.source_relation_sum[source]
            / max(experience.source_exposures[source], 1)
        )
        target_rate = (
            experience.target_relation_sum[target]
            / max(experience.target_exposures[target], 1)
        )
        interaction_only = weights * 0.5 * (source_rate + target_rate)

        tensor_rates = bridge.predict(semantics[source], semantics[target])
        full = weights * tensor_rates

        shuffled_rates = shuffled_bridge.predict(
            shuffled[source], shuffled[target]
        )
        shuffled_scores = weights * shuffled_rates

        for name, scores in [
            ("semantic-only", semantic_only),
            ("interaction-marginal-only", interaction_only),
            ("tensor-no-resolution", tensor_rates),
            ("full-learned-world-model", full),
            ("shuffled-semantic-control", shuffled_scores),
        ]:
            methods[name].append(_tie_hit1(scores, truth))

    return {
        "available": True,
        "fit_samples": len(samples),
        "heldout_internal_pair_collisions": internal_pair_collision,
        "methods": {
            name: float(np.mean(values))
            for name, values in methods.items()
        },
    }


def run_case(
    seed: int,
    *,
    perception_noise: float,
    probes_per_entity: int,
    interactions_per_pair: int = 10,
) -> dict:
    rng = np.random.default_rng(seed)
    centers = perceptual_centers(rng)
    runtime = HybridCortexRuntime(
        embedding_dim=EMBEDDING_DIM,
        max_entities=64,
        budget=24,
        nuisance_mode="identity-only",
    )
    experience = Experience()

    probe_schedule = [
        entity
        for _ in range(probes_per_entity)
        for entity in range(ENTITIES)
    ]
    rng.shuffle(probe_schedule)

    for entity in probe_schedule:
        binding = observe_single(
            runtime,
            experience,
            entity,
            centers,
            rng,
            perception_noise,
        )
        outcome = int(rng.choice(OUTCOMES, p=probe_distribution(entity)))
        experience.add_semantic_outcome(binding, outcome)

    held_true = {(s, t) for s, t, _truth in heldout_true_pairs()}
    interaction_schedule = []
    for source in range(SOURCE_COUNT):
        for target in range(SOURCE_COUNT, ENTITIES):
            if (source, target) in held_true:
                continue
            interaction_schedule.extend(
                [(source, target)] * interactions_per_pair
            )
    rng.shuffle(interaction_schedule)

    checkpoints = {}
    checkpoint_indices = {
        int(round(len(interaction_schedule) * fraction)): fraction
        for fraction in (0.25, 0.50, 1.0)
    }

    for index, (source, target) in enumerate(interaction_schedule, start=1):
        source_binding, target_binding = observe_pair(
            runtime,
            experience,
            source,
            target,
            centers,
            rng,
            perception_noise,
        )
        relation = int(
            rng.choice(
                RELATIONS,
                p=interaction_distribution(source, target),
            )
        )
        experience.add_interaction(
            source_binding,
            target_binding,
            relation,
        )

        if index in checkpoint_indices:
            fraction = checkpoint_indices[index]
            checkpoints[str(fraction)] = checkpoint_prediction(
                experience,
                rng_seed=seed + index * 1009,
            )

    ids = identity_metrics(experience.identity_records)
    semantics = semantic_separation(experience)

    final = checkpoints.get("1.0", {"available": False})
    return {
        "protocol": "learned-world-model-v1",
        "seed": seed,
        "perception_noise": perception_noise,
        "probes_per_entity": probes_per_entity,
        "interactions_per_pair": interactions_per_pair,
        "true_entities": ENTITIES,
        "internal_entities": (
            max(experience.last_binding.values()) + 1
            if experience.last_binding
            else 0
        ),
        "identity": ids,
        "semantic": semantics,
        "checkpoints": checkpoints,
        "final_full_hit_at_1": (
            final["methods"]["full-learned-world-model"]
            if final.get("available")
            else None
        ),
        "final_interaction_only_hit_at_1": (
            final["methods"]["interaction-marginal-only"]
            if final.get("available")
            else None
        ),
        "final_shuffled_hit_at_1": (
            final["methods"]["shuffled-semantic-control"]
            if final.get("available")
            else None
        ),
    }


def campaign(base_seeds: int, sweep_seeds: int, output: Path) -> None:
    rows = []

    for seed in range(base_seeds):
        rows.append(
            run_case(
                seed,
                perception_noise=0.03,
                probes_per_entity=40,
            )
        )

    for perception_noise in (0.03, 0.12, 0.25):
        for probes in (5, 20, 80):
            for seed in range(sweep_seeds):
                rows.append(
                    run_case(
                        10000 + seed,
                        perception_noise=perception_noise,
                        probes_per_entity=probes,
                    )
                )

    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    base = [
        row
        for row in rows
        if row["seed"] < base_seeds
        and row["perception_noise"] == 0.03
        and row["probes_per_entity"] == 40
    ]
    print(
        json.dumps(
            {
                "suite": "base",
                "seeds": len(base),
                "identity_purity": float(
                    np.mean([row["identity"]["purity"] for row in base])
                ),
                "fragmentation": float(
                    np.mean(
                        [row["identity"]["fragmentation"] for row in base]
                    )
                ),
                "semantic_separation": float(
                    np.mean(
                        [
                            row["semantic"]["separation_ratio"]
                            for row in base
                        ]
                    )
                ),
                "hit25": float(
                    np.mean(
                        [
                            row["checkpoints"]["0.25"]["methods"][
                                "full-learned-world-model"
                            ]
                            for row in base
                        ]
                    )
                ),
                "hit50": float(
                    np.mean(
                        [
                            row["checkpoints"]["0.5"]["methods"][
                                "full-learned-world-model"
                            ]
                            for row in base
                        ]
                    )
                ),
                "hit100": float(
                    np.mean([row["final_full_hit_at_1"] for row in base])
                ),
                "interaction_only_hit100": float(
                    np.mean(
                        [
                            row["final_interaction_only_hit_at_1"]
                            for row in base
                        ]
                    )
                ),
                "shuffled_hit100": float(
                    np.mean(
                        [row["final_shuffled_hit_at_1"] for row in base]
                    )
                ),
            },
            sort_keys=True,
        )
    )

    for perception_noise in (0.03, 0.12, 0.25):
        for probes in (5, 20, 80):
            subset = [
                row
                for row in rows
                if row["seed"] >= 10000
                and row["perception_noise"] == perception_noise
                and row["probes_per_entity"] == probes
            ]
            print(
                json.dumps(
                    {
                        "suite": "sweep",
                        "perception_noise": perception_noise,
                        "probes_per_entity": probes,
                        "seeds": len(subset),
                        "identity_purity": float(
                            np.mean(
                                [
                                    row["identity"]["purity"]
                                    for row in subset
                                ]
                            )
                        ),
                        "fragmentation": float(
                            np.mean(
                                [
                                    row["identity"]["fragmentation"]
                                    for row in subset
                                ]
                            )
                        ),
                        "semantic_separation": float(
                            np.mean(
                                [
                                    row["semantic"]["separation_ratio"]
                                    for row in subset
                                ]
                            )
                        ),
                        "full_hit100": float(
                            np.mean(
                                [
                                    row["final_full_hit_at_1"]
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
    parser.add_argument("--base-seeds", type=int, default=12)
    parser.add_argument("--sweep-seeds", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("learned-world-model-v1.jsonl"),
    )
    args = parser.parse_args()
    campaign(args.base_seeds, args.sweep_seeds, args.output)


if __name__ == "__main__":
    main()
