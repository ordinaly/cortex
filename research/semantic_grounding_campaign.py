"""Development/preflight campaign for hybrid semantic-grounding-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

from semantic_grounding import (
    CortexSemanticGrounder,
    NeuralLabelEvidence,
    StructuralSemanticObservation,
)

PROTOCOL = "hybrid-semantic-grounding-v1"
RELATIONS = ("r0", "r1")
STRUCTURE_BUDGETS = (0, 8, 12)
CORRUPTED_PAIRS = (0, 1, 2)
DEVELOPMENT_SEEDS = tuple(range(20))
OFFICIAL_SEEDS = tuple(range(300, 320))

FAMILY_EDGES: dict[str, tuple[tuple[int, str, int], ...]] = {
    "asymmetric-chain": (
        (0, "r0", 1),
        (1, "r1", 2),
        (2, "r0", 3),
        (3, "r1", 4),
        (0, "r1", 3),
        (1, "r0", 4),
    ),
    "asymmetric-branch": (
        (0, "r0", 1),
        (0, "r1", 2),
        (1, "r1", 3),
        (2, "r0", 3),
        (3, "r1", 4),
        (1, "r0", 4),
        (2, "r1", 4),
    ),
    "asymmetric-weave": (
        (0, "r0", 2),
        (2, "r1", 4),
        (0, "r1", 1),
        (1, "r0", 3),
        (3, "r1", 4),
        (1, "r1", 4),
        (2, "r0", 3),
    ),
}

SYMMETRIC_EDGES = (
    (0, "r0", 1),
    (0, "r0", 2),
    (1, "r1", 3),
    (2, "r1", 3),
    (3, "r0", 4),
)


def _family_code(family: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(family))


def build_world(family: str, seed: int):
    if family not in FAMILY_EDGES:
        raise ValueError(f"unknown family: {family}")
    rng = Random(seed * 1009 + _family_code(family))
    concepts = [f"C{seed}_{index}" for index in range(5)]
    labels = [f"L{seed}_{index}" for index in range(5)]
    rng.shuffle(concepts)
    rng.shuffle(labels)

    concept_edges = frozenset(
        (concepts[left], relation, concepts[right])
        for left, relation, right in FAMILY_EDGES[family]
    )
    semantic_edges = frozenset(
        (labels[left], relation, labels[right])
        for left, relation, right in FAMILY_EDGES[family]
    )
    truth = {
        concepts[index]: labels[index]
        for index in range(5)
    }
    return concepts, labels, concept_edges, semantic_edges, truth


def structural_observation_order(family: str, seed: int, world):
    concepts, _labels, concept_edges, _semantic_edges, _truth = world
    observations = [
        StructuralSemanticObservation(
            source=left,
            relation=relation,
            target=right,
            present=(left, relation, right) in concept_edges,
        )
        for left in concepts
        for right in concepts
        if left != right
        for relation in RELATIONS
    ]
    positive = [item for item in observations if item.present]
    negative = [item for item in observations if not item.present]
    rng = Random(seed * 3001 + _family_code(family) + 17)
    rng.shuffle(positive)
    rng.shuffle(negative)

    ordered = []
    p = 0
    n = 0
    while p < len(positive) or n < len(negative):
        if p < len(positive):
            ordered.append(positive[p])
            p += 1
        if n < len(negative):
            ordered.append(negative[n])
            n += 1
    return tuple(ordered)


def neural_distributions(world, seed: int, corrupted_pairs: int):
    concepts, labels, _concept_edges, _semantic_edges, truth = world
    if corrupted_pairs not in CORRUPTED_PAIRS:
        raise ValueError("unsupported corruption level")

    rng = Random(seed * 7001 + corrupted_pairs * 97)
    order = list(concepts)
    rng.shuffle(order)

    wrong: dict[str, str] = {}
    for index in range(corrupted_pairs):
        left = order[2 * index]
        right = order[2 * index + 1]
        wrong[left] = truth[right]
        wrong[right] = truth[left]

    distributions: dict[str, dict[str, float]] = {}
    for concept in concepts:
        true_label = truth[concept]
        if concept in wrong:
            wrong_label = wrong[concept]
            distribution = {label: 0.0 for label in labels}
            distribution[true_label] = 0.08
            distribution[wrong_label] = 0.90
            remaining = [
                label for label in labels
                if label not in {true_label, wrong_label}
            ]
            for label in remaining:
                distribution[label] = 0.02 / len(remaining)
        else:
            distribution = {
                label: 0.16 / (len(labels) - 1)
                for label in labels
            }
            distribution[true_label] = 0.84
        distributions[concept] = distribution
    return distributions


def run_case(
    family: str,
    seed: int,
    corrupted_pairs: int,
    structure_budget: int,
) -> dict:
    world = build_world(family, seed)
    concepts, labels, _concept_edges, semantic_edges, truth = world
    neural = neural_distributions(world, seed, corrupted_pairs)

    grounder = CortexSemanticGrounder(
        concepts,
        labels,
        RELATIONS,
        semantic_edges,
    )
    for concept in concepts:
        grounder.observe_neural(
            NeuralLabelEvidence(
                concept=concept,
                probabilities=neural[concept],
            )
        )

    observations = structural_observation_order(family, seed, world)
    grounder.observe_structure_many(observations[:structure_budget])

    reads = {concept: grounder.read(concept) for concept in concepts}
    neural_top = {
        concept: max(
            neural[concept].items(),
            key=lambda item: item[1],
        )[0]
        for concept in concepts
    }

    resolved = [
        concept for concept in concepts
        if reads[concept].resolved
    ]
    neural_wrong = [
        concept for concept in concepts
        if neural_top[concept] != truth[concept]
    ]
    neural_correct = [
        concept for concept in concepts
        if neural_top[concept] == truth[concept]
    ]
    reinspect = [
        concept for concept in concepts
        if reads[concept].feedback.action == "reinspect"
    ]

    resolved_correct = sum(
        reads[concept].label == truth[concept]
        for concept in resolved
    )
    corrected = sum(
        reads[concept].resolved
        and reads[concept].label == truth[concept]
        for concept in neural_wrong
    )
    false_reinspect = sum(
        reads[concept].feedback.action == "reinspect"
        for concept in neural_correct
    )
    correct_reinspect = sum(
        reads[concept].resolved
        and reads[concept].label == truth[concept]
        and neural_top[concept] != truth[concept]
        for concept in reinspect
    )

    return {
        "protocol": PROTOCOL,
        "case": "normal",
        "family": family,
        "seed": seed,
        "corrupted_pairs": corrupted_pairs,
        "structure_budget": structure_budget,
        "concept_count": len(concepts),
        "label_overlap": len(set(concepts) & set(labels)),
        "neural_accuracy": sum(
            neural_top[concept] == truth[concept]
            for concept in concepts
        ) / len(concepts),
        "cortex_coverage": len(resolved) / len(concepts),
        "cortex_resolved_accuracy": (
            resolved_correct / max(1, len(resolved))
        ),
        "effective_correct_fraction": (
            resolved_correct / len(concepts)
        ),
        "correction_opportunities": len(neural_wrong),
        "correction_rate": corrected / max(1, len(neural_wrong)),
        "false_reinspect_rate": (
            false_reinspect / max(1, len(neural_correct))
        ),
        "reinspect_count": len(reinspect),
        "reinspect_precision": (
            correct_reinspect / max(1, len(reinspect))
        ),
        "mean_resolved_confidence": (
            sum(reads[concept].confidence for concept in resolved)
            / max(1, len(resolved))
        ),
    }


def run_ambiguity_control(seed: int) -> dict:
    concepts = [f"AC{seed}_{index}" for index in range(5)]
    labels = [f"AL{seed}_{index}" for index in range(5)]
    rng = Random(seed * 991)
    rng.shuffle(concepts)
    rng.shuffle(labels)

    concept_edges = frozenset(
        (concepts[left], relation, concepts[right])
        for left, relation, right in SYMMETRIC_EDGES
    )
    semantic_edges = frozenset(
        (labels[left], relation, labels[right])
        for left, relation, right in SYMMETRIC_EDGES
    )
    truth = {
        concepts[index]: labels[index]
        for index in range(5)
    }

    grounder = CortexSemanticGrounder(
        concepts,
        labels,
        RELATIONS,
        semantic_edges,
    )

    for index, concept in enumerate(concepts):
        if index in {1, 2}:
            probabilities = {
                label: 0.005 for label in labels
            }
            probabilities[labels[1]] = 0.4925
            probabilities[labels[2]] = 0.4925
        else:
            probabilities = {
                label: 0.025 for label in labels
            }
            probabilities[labels[index]] = 0.90
        total = sum(probabilities.values())
        probabilities = {
            label: probability / total
            for label, probability in probabilities.items()
        }
        grounder.observe_neural(
            NeuralLabelEvidence(
                concept=concept,
                probabilities=probabilities,
            )
        )

    observations = [
        StructuralSemanticObservation(
            source=left,
            relation=relation,
            target=right,
            present=(left, relation, right) in concept_edges,
        )
        for left in concepts
        for right in concepts
        if left != right
        for relation in RELATIONS
    ]
    grounder.observe_structure_many(observations)

    reads = {concept: grounder.read(concept) for concept in concepts}
    twins = concepts[1:3]
    non_twins = [concepts[0], concepts[3], concepts[4]]

    return {
        "protocol": PROTOCOL,
        "case": "ambiguity",
        "family": "symmetric-twins",
        "seed": seed,
        "label_overlap": len(set(concepts) & set(labels)),
        "twin_unresolved_fraction": sum(
            not reads[concept].resolved
            for concept in twins
        ) / len(twins),
        "non_twin_resolved_accuracy": sum(
            reads[concept].resolved
            and reads[concept].label == truth[concept]
            for concept in non_twins
        ) / len(non_twins),
        "twin_mean_confidence": sum(
            reads[concept].confidence
            for concept in twins
        ) / len(twins),
    }


def campaign(seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for seed in seeds:
        for family in FAMILY_EDGES:
            for corrupted_pairs in CORRUPTED_PAIRS:
                for structure_budget in STRUCTURE_BUDGETS:
                    rows.append(
                        run_case(
                            family,
                            seed,
                            corrupted_pairs,
                            structure_budget,
                        )
                    )
        rows.append(run_ambiguity_control(seed))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--development", action="store_true")
    group.add_argument("--official", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = (
        OFFICIAL_SEEDS
        if args.official
        else DEVELOPMENT_SEEDS
    )
    rows = campaign(seeds)
    args.output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
