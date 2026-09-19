from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from structural_analogy_campaign import (
    ASYMMETRIC,
    FAMILY_EDGES,
    SYMMETRIC,
    TARGET_BUDGETS,
    broken_analogy_control,
    build_world,
    full_identifiability,
    near_isomorphic_control,
    run_budget,
)


def test_source_and_target_namespaces_are_disjoint():
    for family in FAMILY_EDGES:
        world = build_world(family, 2)
        assert set(world.source_nodes).isdisjoint(world.target_nodes)
        assert set(world.source_relations).isdisjoint(
            world.target_relations
        )


def test_full_identifiability_matches_declared_family_symmetry():
    for family in ASYMMETRIC:
        row = full_identifiability(build_world(family, 3))
        assert row["candidate_count"] == 1
        assert row["mapping_unique"]

    for family in SYMMETRIC:
        row = full_identifiability(build_world(family, 3))
        assert row["candidate_count"] > 1
        assert not row["mapping_unique"]
        assert row["entity_ambiguous"] > 0


def test_nested_evidence_never_expands_version_space():
    for family in FAMILY_EDGES:
        world = build_world(family, 4)
        rows = [
            run_budget(world, budget)
            for budget in TARGET_BUDGETS
        ]
        counts = [row["candidate_count"] for row in rows]
        assert all(
            right <= left
            for left, right in zip(counts, counts[1:])
        )
        assert all(count > 0 for count in counts)


def test_resolved_heldout_analogy_predictions_are_exact():
    for family in FAMILY_EDGES:
        world = build_world(family, 5)
        for budget in TARGET_BUDGETS:
            row = run_budget(world, budget)
            metrics = row["metrics"]["cortex"]
            if metrics["resolved"] > 0:
                assert metrics["resolved_accuracy"] == 1.0


def test_symmetric_mapping_can_remain_ambiguous_while_predicting():
    for family in SYMMETRIC:
        world = build_world(family, 6)
        full = full_identifiability(world)
        assert not full["mapping_unique"]
        row = run_budget(world, 36)
        assert row["candidate_count"] > 1
        assert row["metrics"]["cortex"]["resolved"] > 0


def test_near_isomorphic_complete_control_is_rejected():
    for family in ASYMMETRIC:
        row = near_isomorphic_control(build_world(family, 7))
        assert row["rejected"]
        assert row["candidate_count"] == 0


def test_explicit_broken_analogy_contradiction_is_rejected():
    for family in FAMILY_EDGES:
        row = broken_analogy_control(build_world(family, 8))
        assert row["contradiction_found"]
        assert row["before_candidates"] > 0
        assert row["after_candidates"] == 0
        assert row["rejected"]


def test_development_transfer_has_nonzero_heldout_coverage():
    coverages = []
    for family in FAMILY_EDGES:
        row = run_budget(build_world(family, 9), 24)
        coverages.append(row["metrics"]["cortex"]["coverage"])
        assert row["metrics"]["direct-memory"]["coverage"] == 0.0
    assert sum(coverages) / len(coverages) > 0.0
