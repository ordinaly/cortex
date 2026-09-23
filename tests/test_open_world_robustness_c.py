from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from open_world_robustness_c_campaign import run_world
from open_world_robustness_world import build_world


def test_owr_c_separates_relation_schema_from_outcome():
    world = build_world("C", 301)
    assert world.oracle["schema_outcome_separation"] is True
    known = world.oracle["known_predicate"]
    novel = world.oracle["novel_predicate"]
    assert known != novel
    assert all(len(obs.context) == 2 for obs in world.observations)


def test_owr_c_declares_compound_and_ambiguous_cases():
    world = build_world("C", 302)
    expected = world.oracle["expected_components"]
    assert expected["CQ000"] == ("new_regime",)
    assert set(expected["CQ001"]) == {"new_entity_type", "new_relation"}
    assert world.oracle["ambiguous_queries"] == ("CQ002",)
    assert set(world.oracle["ambiguous_competitors"]["CQ002"]) == {
        ("new_entity_type",),
        ("new_regime",),
    }


def test_owr_c_current_surface_and_public_control_are_both_reported():
    result = run_world(build_world("C", 303))
    current = result["current_cortex"]
    public = result["public_component_baseline"]
    assert current["compound_component_output_supported"] is False
    assert current["compound_component_coverage"] == 0.0
    assert 0.0 <= current["local_relation_coverage"] <= 1.0
    assert 0.0 <= current["local_relation_accuracy"] <= 1.0
    assert public["resolved_cases"] == 2
    assert public["ambiguous_cases"] == 1


def test_owr_c_public_fixture_is_discriminable_without_oracle_inputs():
    result = run_world(build_world("C", 304))
    public = result["public_component_baseline"]
    assert public["resolved_exact"] == public["resolved_cases"]
    assert public["ambiguous_unresolved"] == public["ambiguous_cases"]
