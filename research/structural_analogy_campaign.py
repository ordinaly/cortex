"""Official campaign for structural-analogy-v1."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

from structural_analogy import (
    CortexStructuralAnalogy,
    DirectStructuralMemory,
    StructuralObservation,
    TypedEdge,
    all_target_triples,
    full_observations,
    map_edges,
)

SOURCE_RELATIONS = ("R0", "R1", "R2")
TARGET_BUDGETS = (12, 24, 36)
OFFICIAL_SEEDS = tuple(range(300, 320))

FAMILY_EDGES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "asymmetric-chain": (
        (0, 0, 1),
        (1, 1, 2),
        (2, 2, 3),
        (3, 0, 4),
        (4, 1, 5),
        (0, 2, 3),
        (1, 0, 4),
        (2, 1, 5),
    ),
    "asymmetric-branch": (
        (0, 0, 1),
        (0, 1, 2),
        (0, 2, 3),
        (1, 1, 4),
        (2, 2, 4),
        (3, 0, 5),
        (4, 0, 5),
        (1, 2, 5),
        (2, 0, 3),
    ),
    "symmetric-twins": (
        (0, 0, 1),
        (0, 0, 2),
        (1, 1, 3),
        (2, 1, 3),
        (3, 2, 4),
        (4, 0, 5),
    ),
    "symmetric-diamond": (
        (0, 0, 1),
        (0, 0, 2),
        (1, 1, 3),
        (2, 1, 3),
        (3, 2, 4),
        (3, 0, 5),
        (4, 1, 5),
    ),
}

ASYMMETRIC = {"asymmetric-chain", "asymmetric-branch"}
SYMMETRIC = {"symmetric-twins", "symmetric-diamond"}


@dataclass(frozen=True)
class AnalogyWorld:
    family: str
    seed: int
    source_nodes: tuple[str, ...]
    source_relations: tuple[str, ...]
    source_edges: frozenset[TypedEdge]
    target_nodes: tuple[str, ...]
    target_relations: tuple[str, ...]
    target_edges: frozenset[TypedEdge]
    hidden_node_map: dict[str, str]
    hidden_relation_map: dict[str, str]


def _family_code(family: str) -> int:
    return sum((index + 1) * ord(ch) for index, ch in enumerate(family))


def build_world(family: str, seed: int) -> AnalogyWorld:
    if family not in FAMILY_EDGES:
        raise ValueError(f"unknown family: {family}")

    rng = Random(seed * 1009 + _family_code(family))

    source_labels = [f"S{seed}_{index}" for index in range(6)]
    target_labels = [f"T{seed}_{index}" for index in range(6)]
    source_relation_labels = [f"SR{seed}_{index}" for index in range(3)]
    target_relation_labels = [f"TR{seed}_{index}" for index in range(3)]

    rng.shuffle(source_labels)
    rng.shuffle(target_labels)
    rng.shuffle(source_relation_labels)
    rng.shuffle(target_relation_labels)

    canonical_to_source_node = {
        index: source_labels[index]
        for index in range(6)
    }
    canonical_to_target_node = {
        index: target_labels[index]
        for index in range(6)
    }
    canonical_to_source_relation = {
        index: source_relation_labels[index]
        for index in range(3)
    }
    canonical_to_target_relation = {
        index: target_relation_labels[index]
        for index in range(3)
    }

    source_edges = frozenset(
        (
            canonical_to_source_node[left],
            canonical_to_source_relation[relation],
            canonical_to_source_node[right],
        )
        for left, relation, right in FAMILY_EDGES[family]
    )

    node_map = {
        canonical_to_source_node[index]:
        canonical_to_target_node[index]
        for index in range(6)
    }
    relation_map = {
        canonical_to_source_relation[index]:
        canonical_to_target_relation[index]
        for index in range(3)
    }
    target_edges = map_edges(
        source_edges,
        node_map,
        relation_map,
    )

    return AnalogyWorld(
        family=family,
        seed=seed,
        source_nodes=tuple(sorted(source_labels)),
        source_relations=tuple(sorted(source_relation_labels)),
        source_edges=source_edges,
        target_nodes=tuple(sorted(target_labels)),
        target_relations=tuple(sorted(target_relation_labels)),
        target_edges=target_edges,
        hidden_node_map=node_map,
        hidden_relation_map=relation_map,
    )


def observation_order(world: AnalogyWorld) -> tuple[StructuralObservation, ...]:
    rng = Random(world.seed * 5003 + _family_code(world.family) + 17)
    all_obs = list(
        full_observations(
            world.target_nodes,
            world.target_relations,
            world.target_edges,
        )
    )
    positive = [obs for obs in all_obs if obs.present]
    negative = [obs for obs in all_obs if not obs.present]
    rng.shuffle(positive)
    rng.shuffle(negative)

    ordered = []
    p = 0
    n = 0
    while p < len(positive) or n < len(negative):
        if p < len(positive):
            ordered.append(positive[p])
            p += 1
        for _ in range(2):
            if n < len(negative):
                ordered.append(negative[n])
                n += 1
        if p >= len(positive) and n >= len(negative):
            break

    return tuple(ordered)


def make_model(
    world: AnalogyWorld,
    observations: Iterable[StructuralObservation],
) -> CortexStructuralAnalogy:
    model = CortexStructuralAnalogy(
        world.source_nodes,
        world.source_relations,
        world.source_edges,
        world.target_nodes,
        world.target_relations,
    )
    model.observe_many(observations)
    return model


def score_heldout(
    model,
    world: AnalogyWorld,
    observed: set[TypedEdge],
) -> dict:
    total = 0
    resolved = 0
    correct = 0
    positive_total = 0
    positive_resolved = 0
    positive_correct = 0
    negative_total = 0
    negative_resolved = 0
    negative_correct = 0

    for triple in all_target_triples(
        world.target_nodes,
        world.target_relations,
    ):
        if triple in observed:
            continue
        truth = triple in world.target_edges
        total += 1
        if truth:
            positive_total += 1
        else:
            negative_total += 1

        inference = model.infer(*triple)
        if not inference.resolved:
            continue
        resolved += 1
        hit = inference.value is truth
        correct += int(hit)

        if truth:
            positive_resolved += 1
            positive_correct += int(hit)
        else:
            negative_resolved += 1
            negative_correct += int(hit)

    return {
        "total": total,
        "resolved": resolved,
        "correct": correct,
        "accuracy": correct / max(1, total),
        "coverage": resolved / max(1, total),
        "resolved_accuracy": correct / max(1, resolved),
        "positive_total": positive_total,
        "positive_resolved": positive_resolved,
        "positive_resolved_accuracy": (
            positive_correct / max(1, positive_resolved)
        ),
        "negative_total": negative_total,
        "negative_resolved": negative_resolved,
        "negative_resolved_accuracy": (
            negative_correct / max(1, negative_resolved)
        ),
    }


def mapping_metrics(
    model: CortexStructuralAnalogy,
    world: AnalogyWorld,
) -> dict:
    entity_resolved = 0
    entity_correct = 0
    for source in world.source_nodes:
        read = model.entity_correspondence(source)
        if not read.resolved:
            continue
        entity_resolved += 1
        entity_correct += int(
            read.support[0][0]
            == world.hidden_node_map[source]
        )

    relation_resolved = 0
    relation_correct = 0
    for source in world.source_relations:
        read = model.relation_correspondence(source)
        if not read.resolved:
            continue
        relation_resolved += 1
        relation_correct += int(
            read.support[0][0]
            == world.hidden_relation_map[source]
        )

    return {
        "entity_resolved": entity_resolved,
        "entity_correct": entity_correct,
        "entity_precision": (
            entity_correct / max(1, entity_resolved)
        ),
        "relation_resolved": relation_resolved,
        "relation_correct": relation_correct,
        "relation_precision": (
            relation_correct / max(1, relation_resolved)
        ),
    }


def run_budget(
    world: AnalogyWorld,
    budget: int,
) -> dict:
    ordered = observation_order(world)
    evidence = ordered[:budget]
    observed = {obs.triple for obs in evidence}

    model = make_model(world, evidence)
    scratch = DirectStructuralMemory(evidence)

    return {
        "protocol": "structural-analogy-v1",
        "kind": "compatible",
        "family": world.family,
        "seed": world.seed,
        "budget": budget,
        "candidate_count": model.candidate_count,
        "mapping_unique": model.mapping_unique,
        "entity_label_overlap": len(
            set(world.source_nodes).intersection(
                world.target_nodes
            )
        ),
        "relation_label_overlap": len(
            set(world.source_relations).intersection(
                world.target_relations
            )
        ),
        "mapping": mapping_metrics(model, world),
        "metrics": {
            "cortex": score_heldout(
                model,
                world,
                observed,
            ),
            "direct-memory": score_heldout(
                scratch,
                world,
                observed,
            ),
        },
    }


def full_identifiability(world: AnalogyWorld) -> dict:
    model = make_model(
        world,
        full_observations(
            world.target_nodes,
            world.target_relations,
            world.target_edges,
        ),
    )
    entity_ambiguous = sum(
        int(not model.entity_correspondence(source).resolved)
        for source in world.source_nodes
    )
    return {
        "protocol": "structural-analogy-v1",
        "kind": "full-identifiability",
        "family": world.family,
        "seed": world.seed,
        "candidate_count": model.candidate_count,
        "mapping_unique": model.mapping_unique,
        "entity_ambiguous": entity_ambiguous,
        "mapping": mapping_metrics(model, world),
    }


def near_isomorphic_control(world: AnalogyWorld) -> dict:
    if world.family not in ASYMMETRIC:
        raise ValueError("near-isomorphic control requires asymmetric family")

    all_triples = all_target_triples(
        world.target_nodes,
        world.target_relations,
    )
    extra = next(
        triple
        for triple in all_triples
        if triple not in world.target_edges
    )
    perturbed = frozenset(
        set(world.target_edges) | {extra}
    )
    model = make_model(
        world,
        full_observations(
            world.target_nodes,
            world.target_relations,
            perturbed,
        ),
    )
    return {
        "protocol": "structural-analogy-v1",
        "kind": "near-isomorphic",
        "family": world.family,
        "seed": world.seed,
        "candidate_count": model.candidate_count,
        "rejected": model.incompatible,
        "extra_edge": list(extra),
    }


def broken_analogy_control(world: AnalogyWorld) -> dict:
    ordered = observation_order(world)
    evidence = ordered[:24]
    observed = {obs.triple for obs in evidence}
    model = make_model(world, evidence)
    before = model.candidate_count

    contradiction = None
    for triple in all_target_triples(
        world.target_nodes,
        world.target_relations,
    ):
        if triple in observed:
            continue
        inference = model.infer(*triple)
        if inference.resolved:
            contradiction = StructuralObservation(
                source=triple[0],
                relation=triple[1],
                target=triple[2],
                present=not bool(inference.value),
            )
            break

    if contradiction is None:
        return {
            "protocol": "structural-analogy-v1",
            "kind": "broken-analogy",
            "family": world.family,
            "seed": world.seed,
            "before_candidates": before,
            "contradiction_found": False,
            "after_candidates": before,
            "rejected": False,
        }

    model.observe(contradiction)
    return {
        "protocol": "structural-analogy-v1",
        "kind": "broken-analogy",
        "family": world.family,
        "seed": world.seed,
        "before_candidates": before,
        "contradiction_found": True,
        "after_candidates": model.candidate_count,
        "rejected": model.incompatible,
        "contradiction": [
            contradiction.source,
            contradiction.relation,
            contradiction.target,
            contradiction.present,
        ],
    }


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def campaign(
    output: Path,
    *,
    seeds: Iterable[int] = OFFICIAL_SEEDS,
) -> dict:
    seeds = tuple(seeds)
    worlds = [
        build_world(family, seed)
        for seed in seeds
        for family in FAMILY_EDGES
    ]

    rows = [
        run_budget(world, budget)
        for world in worlds
        for budget in TARGET_BUDGETS
    ]
    full = [
        full_identifiability(world)
        for world in worlds
    ]
    near = [
        near_isomorphic_control(world)
        for world in worlds
        if world.family in ASYMMETRIC
    ]
    broken = [
        broken_analogy_control(world)
        for world in worlds
    ]

    artifact_rows = rows + full + near + broken
    output.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in artifact_rows
        ),
        encoding="utf-8",
    )

    candidate_monotone = []
    for world in worlds:
        selected = sorted(
            (
                row
                for row in rows
                if row["seed"] == world.seed
                and row["family"] == world.family
            ),
            key=lambda row: row["budget"],
        )
        counts = [
            row["candidate_count"]
            for row in selected
        ]
        candidate_monotone.append(
            all(
                right <= left
                for left, right in zip(
                    counts,
                    counts[1:],
                )
            )
        )

    coverage_by_budget = {}
    for budget in TARGET_BUDGETS:
        selected = [
            row
            for row in rows
            if row["budget"] == budget
        ]
        coverage_by_budget[str(budget)] = _mean(
            row["metrics"]["cortex"]["coverage"]
            for row in selected
        )

    all_resolved = sum(
        row["metrics"]["cortex"]["resolved"]
        for row in rows
    )
    all_correct = sum(
        row["metrics"]["cortex"]["correct"]
        for row in rows
    )
    positive_resolved = sum(
        row["metrics"]["cortex"]["positive_resolved"]
        for row in rows
    )
    positive_correct = sum(
        int(
            round(
                row["metrics"]["cortex"][
                    "positive_resolved_accuracy"
                ]
                * row["metrics"]["cortex"][
                    "positive_resolved"
                ]
            )
        )
        for row in rows
    )
    negative_resolved = sum(
        row["metrics"]["cortex"]["negative_resolved"]
        for row in rows
    )
    negative_correct = sum(
        int(
            round(
                row["metrics"]["cortex"][
                    "negative_resolved_accuracy"
                ]
                * row["metrics"]["cortex"][
                    "negative_resolved"
                ]
            )
        )
        for row in rows
    )

    asymmetric_full = [
        row for row in full
        if row["family"] in ASYMMETRIC
    ]
    symmetric_full = [
        row for row in full
        if row["family"] in SYMMETRIC
    ]

    asym_resolved = sum(
        row["mapping"]["entity_resolved"]
        + row["mapping"]["relation_resolved"]
        for row in rows
        if row["family"] in ASYMMETRIC
    )
    asym_correct = sum(
        row["mapping"]["entity_correct"]
        + row["mapping"]["relation_correct"]
        for row in rows
        if row["family"] in ASYMMETRIC
    )

    summary = {
        "protocol": "structural-analogy-v1",
        "rows": len(rows),
        "seeds": list(seeds),
        "families": list(FAMILY_EDGES),
        "budgets": list(TARGET_BUDGETS),
        "metrics": {
            "entity_label_overlap_max": max(
                row["entity_label_overlap"]
                for row in rows
            ),
            "relation_label_overlap_max": max(
                row["relation_label_overlap"]
                for row in rows
            ),
            "heldout_resolved_accuracy": (
                all_correct / max(1, all_resolved)
            ),
            "positive_resolved_accuracy": (
                positive_correct / max(1, positive_resolved)
            ),
            "negative_resolved_accuracy": (
                negative_correct / max(1, negative_resolved)
            ),
            "coverage_by_budget": coverage_by_budget,
            "candidate_monotonicity_rate": _mean(
                int(value)
                for value in candidate_monotone
            ),
            "asymmetric_full_unique_rate": _mean(
                int(row["mapping_unique"])
                for row in asymmetric_full
            ),
            "symmetric_full_ambiguity_rate": _mean(
                int(
                    (not row["mapping_unique"])
                    and row["entity_ambiguous"] > 0
                )
                for row in symmetric_full
            ),
            "asymmetric_resolved_mapping_precision": (
                asym_correct / max(1, asym_resolved)
            ),
            "near_isomorphic_rejection_rate": _mean(
                int(row["rejected"])
                for row in near
            ),
            "broken_analogy_rejection_rate": _mean(
                int(
                    row["contradiction_found"]
                    and row["rejected"]
                )
                for row in broken
            ),
            "direct_memory_coverage": _mean(
                row["metrics"]["direct-memory"]["coverage"]
                for row in rows
            ),
        },
        "full_identifiability": full,
        "near_isomorphic": near,
        "broken_analogy": broken,
    }

    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("structural-analogy-v1.jsonl"),
    )
    parser.add_argument(
        "--development-seeds",
        type=int,
        default=0,
        help=(
            "Use development seeds 0..N-1 instead of "
            "official seeds 300..319."
        ),
    )
    args = parser.parse_args()

    seeds = (
        tuple(range(args.development_seeds))
        if args.development_seeds > 0
        else OFFICIAL_SEEDS
    )
    campaign(args.output, seeds=seeds)


if __name__ == "__main__":
    main()
