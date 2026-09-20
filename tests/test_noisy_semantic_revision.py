from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from noisy_semantic_revision_campaign import (
    run_case,
)
from semantic_revision import CortexSemanticRevision
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
