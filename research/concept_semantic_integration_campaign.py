"""Campaign for concept-semantic-integration-v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from concept_semantic_integration import (
    CortexConceptSemanticBridge,
    EntityNeuralEvidence,
    EntityStructuralObservation,
)
from latent_concept_induction import (
    pairwise_partition_f1,
)

PROTOCOL = "concept-semantic-integration-v1"
DEVELOPMENT_SEEDS = tuple(range(30))
OFFICIAL_SEEDS = tuple(range(800, 820))

RELATIONS = ("r0", "r1", "r2")
LABELS = ("A", "B", "C", "D", "E")
SEMANTIC_EDGES = frozenset(
    {
        ("A", "r0", "B"),
        ("B", "r1", "C"),
        ("A", "r2", "C"),
        ("C", "r0", "A"),
    }
)

THREE_TRUTH = (0, 0, 1, 1, 2, 2)
PROBABILITIES = np.asarray(
    [
        [0.60, 0.20, 0.08, 0.05, 0.04, 0.03],
        [0.04, 0.08, 0.60, 0.18, 0.05, 0.05],
        [0.03, 0.03, 0.04, 0.15, 0.30, 0.45],
    ],
    dtype=float,
)


def _family_code(name: str) -> int:
    return sum(
        (index + 1) * ord(char)
        for index, char in enumerate(name)
    )


def _rng(seed: int, family: str) -> np.random.Generator:
    return np.random.default_rng(
        seed * 130363 + _family_code(family)
    )


def _counts(
    seed: int,
    family: str,
    *,
    per_entity: int = 30,
) -> tuple[np.ndarray, tuple[int, ...]]:
    rng = _rng(seed, family)
    values = np.asarray(
        [
            rng.multinomial(
                per_entity,
                PROBABILITIES[THREE_TRUTH[entity]],
            )
            for entity in range(6)
        ],
        dtype=float,
    )
    order = rng.permutation(6)
    outcomes = rng.permutation(6)
    return (
        values[order][:, outcomes],
        tuple(
            THREE_TRUTH[int(index)]
            for index in order
        ),
    )


def _members_to_truth(
    members: frozenset[int],
    truth: tuple[int, ...],
) -> int | None:
    groups = {truth[index] for index in members}
    if len(groups) != 1:
        return None
    return next(iter(groups))


def _partition_from_active(
    active,
    entity_count: int,
) -> tuple[int, ...]:
    labels = [-1] * entity_count
    next_group = 0
    for concept in active:
        for entity in concept.members:
            labels[entity] = next_group
        next_group += 1
    for entity, value in enumerate(labels):
        if value < 0:
            labels[entity] = next_group
            next_group += 1
    return tuple(labels)


def _distribution(
    top: str,
    *,
    true_label: str,
    adversarial: bool,
) -> dict[str, float]:
    result = {label: 0.02 for label in LABELS}
    result[top] = 0.82
    if adversarial and true_label != top:
        result[true_label] = 0.10
    else:
        result[true_label] = max(
            result[true_label],
            0.10,
        )
    total = sum(result.values())
    return {
        label: value / total
        for label, value in result.items()
    }


def _entity_neural(
    truth: tuple[int, ...],
    group_labels: dict[int, str],
    *,
    swap_first_two: bool,
) -> list[EntityNeuralEvidence]:
    evidence = []
    for entity, group in enumerate(truth):
        true_label = group_labels[group]
        top = true_label
        if swap_first_two:
            if group == 0:
                top = group_labels[1]
            elif group == 1:
                top = group_labels[0]
        evidence.append(
            EntityNeuralEvidence(
                entity=entity,
                probabilities=_distribution(
                    top,
                    true_label=true_label,
                    adversarial=(top != true_label),
                ),
                reliability=1.0,
            )
        )
    return evidence


def _entity_structure(
    truth: tuple[int, ...],
    group_labels: dict[int, str],
    semantic_edges=SEMANTIC_EDGES,
) -> list[EntityStructuralObservation]:
    observations = []
    for source in range(len(truth)):
        for target in range(len(truth)):
            if source == target:
                continue
            source_group = truth[source]
            target_group = truth[target]
            if source_group == target_group:
                continue
            source_label = group_labels[source_group]
            target_label = group_labels[target_group]
            for relation in RELATIONS:
                observations.append(
                    EntityStructuralObservation(
                        source=source,
                        relation=relation,
                        target=target,
                        present=(
                            source_label,
                            relation,
                            target_label,
                        ) in semantic_edges,
                        reliability=1.0,
                    )
                )
    return observations


def adversarial_grounding(seed: int) -> dict:
    counts, truth = _counts(
        seed,
        "adversarial-grounding",
    )
    group_labels = {0: "A", 1: "B", 2: "C"}
    neural = _entity_neural(
        truth,
        group_labels,
        swap_first_two=True,
    )
    structural = _entity_structure(
        truth,
        group_labels,
    )

    bridge = CortexConceptSemanticBridge(
        6,
        LABELS,
        RELATIONS,
        SEMANTIC_EDGES,
    )
    read = bridge.advance(
        0,
        counts,
        neural=neural,
        structural=structural,
    )

    predicted_partition = _partition_from_active(
        read.active_concepts,
        6,
    )
    concept_f1 = pairwise_partition_f1(
        predicted_partition,
        truth,
    )

    true_by_id = {}
    for concept in read.active_concepts:
        group = _members_to_truth(
            concept.members,
            truth,
        )
        if group is not None:
            true_by_id[concept.concept_id] = (
                group_labels[group]
            )

    semantic_by_id = read.semantic_by_id()
    resolved = [
        concept_id
        for concept_id in true_by_id
        if semantic_by_id[concept_id].resolved
    ]
    correct = [
        concept_id
        for concept_id in resolved
        if semantic_by_id[concept_id].label
        == true_by_id[concept_id]
    ]

    neural_group_top = {
        group: (
            group_labels[1]
            if group == 0
            else (
                group_labels[0]
                if group == 1
                else group_labels[2]
            )
        )
        for group in range(3)
    }
    raw_correct = mean_bool(
        neural_group_top[group] == group_labels[group]
        for group in range(3)
    )

    distractor_resolved = sum(
        int(
            semantic_by_id[concept_id].resolved
            and semantic_by_id[concept_id].label in {"D", "E"}
        )
        for concept_id in semantic_by_id
    )

    return {
        "protocol": PROTOCOL,
        "case": "adversarial-grounding",
        "seed": seed,
        "concept_f1": concept_f1,
        "active_concepts": len(read.active_concepts),
        "raw_neural_accuracy": raw_correct,
        "semantic_coverage": len(resolved) / 3,
        "semantic_accuracy": (
            len(correct) / max(1, len(resolved))
        ),
        "effective_semantic_accuracy": len(correct) / 3,
        "distractor_resolved": distractor_resolved,
    }


def mean_bool(values) -> float:
    values = list(values)
    return sum(int(value) for value in values) / len(values)


def label_isolation(seed: int) -> dict:
    counts, truth = _counts(
        seed,
        "label-isolation",
    )

    bridge_a = CortexConceptSemanticBridge(
        6,
        ("A", "B", "C", "D", "E"),
        RELATIONS,
        SEMANTIC_EDGES,
    )
    read_a = bridge_a.advance(0, counts)

    renamed_labels = ("Q", "W", "X", "Y", "Z")
    renamed_edges = frozenset(
        {
            ("Q", "r0", "W"),
            ("W", "r1", "X"),
            ("Q", "r2", "X"),
            ("X", "r0", "Q"),
        }
    )
    bridge_b = CortexConceptSemanticBridge(
        6,
        renamed_labels,
        RELATIONS,
        renamed_edges,
    )
    read_b = bridge_b.advance(0, counts)

    members_a = sorted(
        (
            tuple(sorted(concept.members)),
            concept.concept_id,
        )
        for concept in read_a.active_concepts
    )
    members_b = sorted(
        (
            tuple(sorted(concept.members)),
            concept.concept_id,
        )
        for concept in read_b.active_concepts
    )

    return {
        "protocol": PROTOCOL,
        "case": "label-isolation",
        "seed": seed,
        "concept_f1": pairwise_partition_f1(
            _partition_from_active(
                read_a.active_concepts,
                6,
            ),
            truth,
        ),
        "members_equal": members_a == members_b,
        "concept_ids_equal": (
            [item[1] for item in members_a]
            == [item[1] for item in members_b]
        ),
    }


def unnamed_concept(seed: int) -> dict:
    counts, truth = _counts(
        seed,
        "unnamed-concept",
        per_entity=35,
    )
    group_labels = {0: "A", 1: "B", 2: None}

    neural = []
    for entity, group in enumerate(truth):
        if group == 0:
            probabilities = {
                "A": 0.88,
                "B": 0.03,
                "C": 0.03,
                "D": 0.03,
                "E": 0.03,
            }
        elif group == 1:
            probabilities = {
                "A": 0.03,
                "B": 0.88,
                "C": 0.03,
                "D": 0.03,
                "E": 0.03,
            }
        else:
            probabilities = {
                "A": 0.02,
                "B": 0.02,
                "C": 0.32,
                "D": 0.32,
                "E": 0.32,
            }
        neural.append(
            EntityNeuralEvidence(
                entity=entity,
                probabilities=probabilities,
            )
        )

    semantic_edges = frozenset(
        {("A", "r0", "B")}
    )
    structural = []
    for source in range(6):
        for target in range(6):
            if source == target:
                continue
            if truth[source] == truth[target]:
                continue
            source_label = group_labels[truth[source]]
            target_label = group_labels[truth[target]]
            for relation in RELATIONS:
                present = (
                    source_label is not None
                    and target_label is not None
                    and (
                        source_label,
                        relation,
                        target_label,
                    ) in semantic_edges
                )
                structural.append(
                    EntityStructuralObservation(
                        source=source,
                        relation=relation,
                        target=target,
                        present=present,
                    )
                )

    bridge = CortexConceptSemanticBridge(
        6,
        LABELS,
        RELATIONS,
        semantic_edges,
    )
    read = bridge.advance(
        0,
        counts,
        neural=neural,
        structural=structural,
    )

    known_correct = 0
    known_total = 0
    unnamed_resolved = None
    unnamed_confidence = None

    semantic_by_id = read.semantic_by_id()
    for concept in read.active_concepts:
        group = _members_to_truth(
            concept.members,
            truth,
        )
        if group is None:
            continue
        semantic = semantic_by_id[concept.concept_id]
        if group in (0, 1):
            known_total += 1
            known_correct += int(
                semantic.resolved
                and semantic.label
                == group_labels[group]
            )
        else:
            unnamed_resolved = semantic.resolved
            unnamed_confidence = semantic.confidence

    return {
        "protocol": PROTOCOL,
        "case": "unnamed-concept",
        "seed": seed,
        "concept_f1": pairwise_partition_f1(
            _partition_from_active(
                read.active_concepts,
                6,
            ),
            truth,
        ),
        "known_semantic_accuracy": (
            known_correct / max(1, known_total)
        ),
        "unnamed_resolved": bool(unnamed_resolved),
        "unnamed_confidence": float(
            unnamed_confidence
            if unnamed_confidence is not None
            else 1.0
        ),
    }


def _lifecycle_neural(
    phase: str,
) -> list[EntityNeuralEvidence]:
    labels = ("broad", "left", "right", "other")
    evidence = []
    for entity in range(6):
        if phase in {"coarse", "merge"}:
            top = "broad"
        elif phase == "split":
            top = "left" if entity < 3 else "right"
        else:
            top = "other"
        probabilities = {
            label: 0.02 for label in labels
        }
        probabilities[top] = 0.94
        total = sum(probabilities.values())
        evidence.append(
            EntityNeuralEvidence(
                entity=entity,
                probabilities={
                    label: value / total
                    for label, value in probabilities.items()
                },
            )
        )
    return evidence


def lifecycle_semantics(seed: int) -> dict:
    rng = _rng(seed, "semantic-lifecycle")
    entity_order = rng.permutation(6)
    outcome_order = rng.permutation(6)

    coarse = np.tile(
        np.asarray([20, 10, 5, 3, 1, 1], dtype=float),
        (6, 1),
    )
    split = np.vstack(
        [
            np.asarray([28, 5, 3, 2, 1, 1], dtype=float)
            for _ in range(3)
        ]
        + [
            np.asarray([1, 1, 2, 3, 5, 28], dtype=float)
            for _ in range(3)
        ]
    )
    unique = []
    for entity in range(6):
        vector = np.full(6, 2.0)
        vector[entity] = 30.0
        unique.append(vector)
    unique = np.asarray(unique)

    labels = ("broad", "left", "right", "other")
    bridge = CortexConceptSemanticBridge(
        6,
        labels,
        tuple(),
        tuple(),
    )

    def permute_evidence(evidence):
        inverse = {
            original: observed
            for observed, original in enumerate(entity_order)
        }
        return [
            EntityNeuralEvidence(
                entity=inverse[item.entity],
                probabilities=item.probabilities,
                reliability=item.reliability,
            )
            for item in evidence
        ]

    snapshots = [
        ("coarse", coarse),
        ("split", split),
        ("merge", coarse),
        ("retract", unique),
    ]
    reads = []
    for epoch, (phase, counts) in enumerate(snapshots):
        reads.append(
            bridge.advance(
                epoch,
                counts[entity_order][:, outcome_order],
                neural=permute_evidence(
                    _lifecycle_neural(phase)
                ),
            )
        )

    split_lineage = [
        event for event in reads[1].semantic_lineage
        if event.kind == "split"
    ]
    merge_lineage = [
        event for event in reads[2].semantic_lineage
        if event.kind == "merge"
    ]
    retract_lineage = [
        event for event in reads[3].semantic_lineage
        if event.kind == "retracted"
    ]

    split_ok = (
        len(split_lineage) == 1
        and split_lineage[0].source_labels
        == ("broad",)
        and set(split_lineage[0].target_labels)
        == {"left", "right"}
    )
    merge_ok = (
        len(merge_lineage) == 1
        and set(merge_lineage[0].source_labels)
        == {"left", "right"}
        and merge_lineage[0].target_labels
        == ("broad",)
    )
    retract_ok = (
        len(retract_lineage) == 1
        and retract_lineage[0].source_labels
        == ("broad",)
        and retract_lineage[0].target_labels
        == tuple()
        and reads[3].semantics == tuple()
    )

    source_ids = (
        split_lineage[0].source_ids
        if split_lineage else tuple()
    )
    split_target_ids = (
        split_lineage[0].target_ids
        if split_lineage else tuple()
    )
    merge_target_ids = (
        merge_lineage[0].target_ids
        if merge_lineage else tuple()
    )
    identity_ok = (
        bool(source_ids)
        and bool(split_target_ids)
        and bool(merge_target_ids)
        and not set(source_ids) & set(split_target_ids)
        and not set(split_target_ids) & set(merge_target_ids)
        and not set(source_ids) & set(merge_target_ids)
    )

    return {
        "protocol": PROTOCOL,
        "case": "semantic-lifecycle",
        "seed": seed,
        "split_semantics_correct": split_ok,
        "merge_semantics_correct": merge_ok,
        "retraction_invalidates_semantics": retract_ok,
        "identity_lineage_correct": identity_ok,
    }


def campaign(seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for seed in seeds:
        rows.append(adversarial_grounding(seed))
        rows.append(label_isolation(seed))
        rows.append(unnamed_concept(seed))
        rows.append(lifecycle_semantics(seed))
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
