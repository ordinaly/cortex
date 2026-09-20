from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from noisy_semantic_revision_campaign import (
    run_case,
)
from semantic_revision import CortexSemanticRevision
from semantic_grounding import NeuralLabelEvidence
from semantic_grounding_campaign import RELATIONS, build_world
from noisy_semantic_revision_campaign import noisy_neural, noisy_structure


def test_noisy_case_is_deterministic():
    left = run_case("asymmetric-chain", 3)
    right = run_case("asymmetric-chain", 3)
    assert left == right


def test_noisy_case_keeps_history_bounded():
    row = run_case("asymmetric-branch", 4)
    assert row["bounded_neural_history"]
    assert row["bounded_structural_history"]


def test_revision_read_exposes_confidence_and_trust():
    concepts, labels, _edges, semantic_edges, truth = build_world(
        "asymmetric-weave",
        5,
    )
    concepts = list(concepts)
    labels = list(labels)
    model = CortexSemanticRevision(
        concepts,
        labels,
        RELATIONS,
        semantic_edges,
    )
    neural, _ = noisy_neural(
        concepts,
        labels,
        truth,
        seed=5,
        family="asymmetric-weave",
        epoch=0,
        error_rate=0.0,
        missing_rate=0.0,
    )
    structural = noisy_structure(
        concepts,
        truth,
        semantic_edges,
        seed=5,
        family="asymmetric-weave",
        epoch=0,
        observation_rate=1.0,
        flip_rate=0.0,
    )
    model.advance(0, neural=neural, structural=structural)
    for concept in concepts:
        read = model.read(concept)
        assert 0.0 <= read.accepted_label_support <= 1.0
        assert 0.0 <= read.challenger_support <= 1.0
        assert read.revision_pressure >= 0.0


def test_noisy_campaign_has_nontrivial_neural_errors():
    row = run_case("asymmetric-chain", 6)
    assert row["neural_top_accuracy"] < 0.95
    assert row["neural_top_accuracy"] > 0.50


def test_current_perceptual_conflict_suspends_trust_before_revision():
    concepts, labels, _edges, semantic_edges, truth = build_world(
        "asymmetric-chain",
        8,
    )
    concepts = list(concepts)
    labels = list(labels)
    model = CortexSemanticRevision(
        concepts,
        labels,
        RELATIONS,
        semantic_edges,
        challenge_patience=3,
    )

    for epoch in range(3):
        neural, _ = noisy_neural(
            concepts,
            labels,
            truth,
            seed=8,
            family="asymmetric-chain",
            epoch=epoch,
            error_rate=0.0,
            missing_rate=0.0,
        )
        structural = noisy_structure(
            concepts,
            truth,
            semantic_edges,
            seed=8,
            family="asymmetric-chain",
            epoch=epoch,
            observation_rate=1.0,
            flip_rate=0.0,
        )
        model.advance(epoch, neural=neural, structural=structural)

    target = concepts[0]
    incumbent = model.read(target).accepted_label
    assert incumbent == truth[target]

    wrong = next(label for label in labels if label != incumbent)
    rest = 0.05 / (len(labels) - 1)
    probabilities = {label: rest for label in labels}
    probabilities[wrong] = 0.95

    model.advance(
        3,
        neural=[
            NeuralLabelEvidence(
                concept=target,
                probabilities=probabilities,
                reliability=1.0,
            )
        ],
    )

    read = model.read(target)
    assert read.accepted_label == incumbent
    assert read.perceptual_conflict
    assert not read.trusted
