from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from open_world_robustness_a_campaign import (
    _dependency_groups,
    run_world,
)
from open_world_robustness_world import build_world


def test_owr_a_reporting_order_is_not_fixed_by_source_identity():
    world = build_world("A", 101)
    blocks = [
        tuple(obs.source_id for obs in world.observations[index:index + 5])
        for index in range(0, min(300, len(world.observations)), 5)
    ]
    assert len(set(blocks)) > 4


def test_owr_a_uses_fresh_transfer_targets_after_calibration():
    world = build_world("A", 102)
    calibration = {
        tuple(value) for value in world.oracle["calibration_contexts"]
    }
    targets = {
        tuple(value) for value in world.oracle["target_contexts"]
    }
    assert calibration.isdisjoint(targets)
    assert world.oracle["target_contexts_seen_during_calibration"] is False
    assert world.oracle["drift"]["after_round"] == 8
    assert world.oracle["learner_receives_source_reliability"] is False
    assert world.oracle["learner_receives_copy_graph"] is False
    assert len(world.oracle["trap_contexts"]) == 4
    assert len(world.oracle["control_contexts"]) == 2


def test_owr_a_target_trap_has_surface_wrong_majority():
    world = build_world("A", 103)
    traps = {tuple(value) for value in world.oracle["trap_contexts"]}
    truth = world.oracle["truth_by_context"]
    for context in traps:
        counts = {}
        for observation in world.observations:
            if observation.context != context:
                continue
            counts[observation.observed_relation] = (
                counts.get(observation.observed_relation, 0) + 1
            )
        expected = truth["|".join(context)]
        winner = max(counts, key=counts.get)
        assert winner != expected


def test_owr_a_public_dependence_signal_recovers_copy_cluster():
    world = build_world("A", 104)
    calibration_query = next(
        query for query in world.queries if query.query_id.startswith("AQC")
    )
    groups = _dependency_groups(world, calibration_query.query_time)
    assert any(len(group) >= 3 for group in groups)
    assert sum(len(group) for group in groups) == 5


def test_owr_a_characterization_has_declared_controls_and_partitions():
    result = run_world(build_world("A", 105))
    expected = {
        "current_cortex_source_blind",
        "global_majority",
        "recent_five_vote",
        "public_dependency_collapsed_vote",
        "oracle_independent_source_reference",
    }
    assert set(result["metrics"]) == expected
    partitions = {
        "all",
        "calibration",
        "target",
        "target_trap",
        "target_control",
    }
    for method in expected:
        assert set(result["metrics"][method]) == partitions
        for partition in result["metrics"][method].values():
            assert 0.0 <= partition["coverage"] <= 1.0
            assert 0.0 <= partition["resolved_accuracy"] <= 1.0
            assert 0.0 <= partition["accuracy"] <= 1.0
            assert 0.0 <= partition["selective_risk"] <= 1.0
