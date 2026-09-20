from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from concept_semantic_integration import (
    CortexConceptSemanticBridge,
    EntityNeuralEvidence,
)
from concept_semantic_integration_campaign import (
    LABELS,
    RELATIONS,
    SEMANTIC_EDGES,
    adversarial_grounding,
    label_isolation,
    lifecycle_semantics,
    unnamed_concept,
)
from semantic_grounding import (
    CortexSemanticGrounder,
    NeuralLabelEvidence,
)


def test_semantic_grounder_allows_surplus_vocabulary():
    model = CortexSemanticGrounder(
        ("Z1", "Z2"),
        ("A", "B", "C"),
        tuple(),
        tuple(),
    )
    model.observe_neural(
        NeuralLabelEvidence(
            concept="Z1",
            probabilities={
                "A": 0.90,
                "B": 0.05,
                "C": 0.05,
            },
        )
    )
    model.observe_neural(
        NeuralLabelEvidence(
            concept="Z2",
            probabilities={
                "A": 0.05,
                "B": 0.90,
                "C": 0.05,
            },
        )
    )
    assert model.read("Z1").label == "A"
    assert model.read("Z2").label == "B"


def test_labels_do_not_change_discovered_concepts():
    row = label_isolation(3)
    assert row["concept_f1"] == 1.0
    assert row["members_equal"]
    assert row["concept_ids_equal"]


def test_adversarial_labels_are_corrected_after_discovery():
    row = adversarial_grounding(4)
    assert row["concept_f1"] == 1.0
    assert row["raw_neural_accuracy"] < 0.5
    assert row["effective_semantic_accuracy"] >= 2.0 / 3.0
    assert row["distractor_resolved"] == 0


def test_resolved_concept_can_remain_unnamed():
    row = unnamed_concept(5)
    assert row["concept_f1"] == 1.0
    assert row["known_semantic_accuracy"] == 1.0
    assert not row["unnamed_resolved"]


def test_semantic_lineage_follows_split_merge_retract():
    row = lifecycle_semantics(6)
    assert row["split_semantics_correct"]
    assert row["merge_semantics_correct"]
    assert row["retraction_invalidates_semantics"]
    assert row["identity_lineage_correct"]


def test_retraction_leaves_no_semantic_records():
    labels = ("broad", "left", "right", "other")
    bridge = CortexConceptSemanticBridge(
        6,
        labels,
        tuple(),
        tuple(),
    )
    coarse = np.tile(
        np.asarray([20, 10, 5, 3, 1, 1], dtype=float),
        (6, 1),
    )
    unique = []
    for entity in range(6):
        vector = np.full(6, 2.0)
        vector[entity] = 30.0
        unique.append(vector)
    unique = np.asarray(unique)

    probabilities = {
        "broad": 0.94,
        "left": 0.02,
        "right": 0.02,
        "other": 0.02,
    }
    first = bridge.advance(
        0,
        coarse,
        neural=[
            EntityNeuralEvidence(
                entity=entity,
                probabilities=probabilities,
            )
            for entity in range(6)
        ],
    )
    assert len(first.semantics) == 1
    assert first.semantics[0].label == "broad"

    second = bridge.advance(1, unique)
    assert second.active_concepts == tuple()
    assert second.semantics == tuple()
