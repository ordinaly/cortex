from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from open_world_robustness_a_campaign import run_world
from open_world_robustness_world import build_world


def test_owr_a_reporting_order_is_not_fixed_by_source_identity():
    world = build_world("A", 101)
    blocks = [
        tuple(obs.source_id for obs in world.observations[index:index + 5])
        for index in range(0, min(300, len(world.observations)), 5)
    ]
    assert len(set(blocks)) > 4


def test_owr_a_has_late_source_drift_and_copied_majority_trap():
    world = build_world("A", 102)
    assert world.oracle["drift"]["after_round"] == 10
    assert world.oracle["copied_majority_trap"]["after_round"] == 14
    assert world.oracle["learner_receives_source_reliability"] is False
    assert world.oracle["learner_receives_copy_graph"] is False
    assert len(world.oracle["trap_contexts"]) == 4


def test_owr_a_current_cortex_characterization_has_declared_controls():
    result = run_world(build_world("A", 103))
    expected = {
        "current_cortex_source_blind",
        "global_majority",
        "recent_five_vote",
        "oracle_independent_source_reference",
    }
    assert set(result["metrics"]) == expected
    for method in expected:
        assert set(result["metrics"][method]) == {
            "all",
            "pre",
            "final",
            "final_trap",
            "final_control",
        }
        for partition in result["metrics"][method].values():
            assert 0.0 <= partition["coverage"] <= 1.0
            assert 0.0 <= partition["resolved_accuracy"] <= 1.0
            assert 0.0 <= partition["accuracy"] <= 1.0
            assert 0.0 <= partition["selective_risk"] <= 1.0
