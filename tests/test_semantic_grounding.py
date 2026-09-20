from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from semantic_grounding import (
    CortexSemanticGrounder,
    NeuralLabelEvidence,
    StructuralSemanticObservation,
)
from semantic_grounding_campaign import (
    RELATIONS,
    build_world,
    neural_distributions,
    run_ambiguity_control,
    run_case,
    structural_observation_order,
)


def test_neural_distribution_validation_is_strict():
    grounder = CortexSemanticGrounder(
        ["c0", "c1"],
        ["l0", "l1"],
        ["r"],
        {("l0", "r", "l1")},
    )
    with pytest.raises(ValueError):
        grounder.observe_neural(
            NeuralLabelEvidence(
                concept="c0",
                probabilities={"l0": 1.0},
            )
        )
    with pytest.raises(ValueError):
        grounder.observe_neural(
            NeuralLabelEvidence(
                concept="c0",
                probabilities={"l0": 0.8, "l1": 0.8},
            )
        )


def test_structure_can_demote_high_confidence_neural_swap():
    concepts = ["c0", "c1", "c2", "c3"]
    labels = ["l0", "l1", "l2", "l3"]
    grounder = CortexSemanticGrounder(
        concepts,
        labels,
        ["r"],
        {
            ("l0", "r", "l1"),
            ("l1", "r", "l2"),
            ("l2", "r", "l3"),
        },
    )

    neural = [
        {"l0": 0.15, "l1": 0.75, "l2": 0.05, "l3": 0.05},
        {"l0": 0.75, "l1": 0.15, "l2": 0.05, "l3": 0.05},
        {"l0": 0.05, "l1": 0.05, "l2": 0.85, "l3": 0.05},
        {"l0": 0.05, "l1": 0.05, "l2": 0.05, "l3": 0.85},
    ]
    for concept, probabilities in zip(concepts, neural):
        grounder.observe_neural(
            NeuralLabelEvidence(concept, probabilities)
        )
    for index in range(3):
        grounder.observe_structure(
            StructuralSemanticObservation(
                f"c{index}",
                "r",
                f"c{index + 1}",
                True,
            )
        )

    first = grounder.read("c0")
    second = grounder.read("c1")
    assert first.resolved and first.label == "l0"
    assert second.resolved and second.label == "l1"
    assert first.neural_top_label == "l1"
    assert second.neural_top_label == "l0"
    assert first.feedback.action == "reinspect"
    assert first.feedback.promote_label == "l0"
    assert first.feedback.demote_label == "l1"


def test_without_structure_grounder_preserves_neural_swap():
    row = run_case(
        "asymmetric-chain",
        seed=3,
        corrupted_pairs=1,
        structure_budget=0,
    )
    assert row["neural_accuracy"] == 0.6
    assert row["cortex_coverage"] == 1.0
    assert row["cortex_resolved_accuracy"] == 0.6
    assert row["correction_rate"] == 0.0


def test_structural_evidence_improves_same_neural_fixture():
    baseline = run_case(
        "asymmetric-chain",
        seed=4,
        corrupted_pairs=1,
        structure_budget=0,
    )
    grounded = run_case(
        "asymmetric-chain",
        seed=4,
        corrupted_pairs=1,
        structure_budget=12,
    )
    assert grounded["effective_correct_fraction"] > baseline[
        "effective_correct_fraction"
    ]
    assert grounded["false_reinspect_rate"] == 0.0


def test_symmetric_semantics_remain_unresolved():
    row = run_ambiguity_control(seed=5)
    assert row["twin_unresolved_fraction"] == 1.0
    assert row["non_twin_resolved_accuracy"] == 1.0
    assert row["twin_mean_confidence"] == pytest.approx(0.5, abs=1.0e-9)


def test_concept_identity_is_independent_of_accepted_label():
    family = "asymmetric-branch"
    seed = 6
    world = build_world(family, seed)
    concepts, labels, _concept_edges, semantic_edges, truth = world
    neural = neural_distributions(world, seed, corrupted_pairs=1)

    grounder = CortexSemanticGrounder(
        concepts,
        labels,
        RELATIONS,
        semantic_edges,
    )
    for concept in concepts:
        grounder.observe_neural(
            NeuralLabelEvidence(concept, neural[concept])
        )

    before = {
        concept: grounder.read(concept).label
        for concept in concepts
    }
    observations = structural_observation_order(
        family,
        seed,
        world,
    )
    grounder.observe_structure_many(observations[:12])
    after = {
        concept: grounder.read(concept).label
        for concept in concepts
    }

    assert set(before) == set(after) == set(concepts)
    assert any(before[concept] != after[concept] for concept in concepts)
    assert all(
        after[concept] == truth[concept]
        for concept in concepts
        if after[concept] is not None
    )


def test_clean_neural_labels_are_not_needlessly_demoted():
    row = run_case(
        "asymmetric-weave",
        seed=7,
        corrupted_pairs=0,
        structure_budget=12,
    )
    assert row["cortex_resolved_accuracy"] == 1.0
    assert row["cortex_coverage"] == 1.0
    assert row["false_reinspect_rate"] == 0.0
