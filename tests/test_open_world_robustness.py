from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from open_world_robustness_world import (
    PROTOCOL,
    STAGES,
    build_suite,
    build_world,
    derive_seed,
    suite_integrity,
    validate_official_material,
)


def test_seed_derivation_is_stable_and_stage_separated():
    left = derive_seed("material", "A", 0)
    assert left == derive_seed("material", "A", 0)
    assert left != derive_seed("material", "B", 0)
    assert left != derive_seed("material", "A", 1)


def test_official_material_requires_merge_sha_shape():
    sha = "a" * 40
    assert validate_official_material(sha) == sha
    for bad in ("dev", "a" * 39, "g" * 40):
        try:
            validate_official_material(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid official material: {bad!r}")


def test_all_stage_worlds_are_deterministic():
    for stage in STAGES:
        left = build_world(stage, 12345)
        right = build_world(stage, 12345)
        assert left.protocol == PROTOCOL
        assert left.fingerprint() == right.fingerprint()
        assert left.oracle_fingerprint() == right.oracle_fingerprint()


def test_stage_a_hides_source_reliability_from_observations():
    world = build_world("A", 11)
    assert world.oracle["learner_receives_source_reliability"] is False
    assert world.oracle["copy_parent"]
    assert world.oracle["drift"]
    for observation in world.observations:
        assert "reliability" not in observation.public_dict()
        assert observation.source_id


def test_stage_b_contains_delay_missingness_duplicates_and_recurrence():
    world = build_world("B", 12)
    assert world.oracle["missing_events"] > 0
    assert world.oracle["delayed_events"] > 0
    assert world.oracle["duplicate_events"] > 0
    schedule = world.oracle["regime_schedule"]
    assert schedule[0][2] == schedule[-1][2]
    assert any(obs.arrival_time > obs.event_time for obs in world.observations)
    assert any(obs.observed_relation is None for obs in world.observations)
    assert any(obs.duplicate_of is not None for obs in world.observations)


def test_stage_c_requires_compound_attribution_and_preserves_ambiguity():
    world = build_world("C", 13)
    components = world.oracle["expected_components"]
    assert len(components["CQ000"]) > 1
    assert len(components["CQ001"]) > 1
    assert world.oracle["ambiguous_queries"] == ("CQ002",)
    ambiguous = next(q for q in world.queries if q.query_id == "CQ002")
    assert not ambiguous.answerable
    assert ambiguous.expected_relation is None


def test_stage_d_is_mostly_distractor_and_has_reactivation_query():
    world = build_world("D", 14)
    assert world.oracle["stream_length"] == 3_000
    salient = sum("salient" in obs.tags for obs in world.observations)
    assert salient / len(world.observations) < 0.06
    assert any(q.kind == "regime-reactivation" for q in world.queries)


def test_stage_e_heldout_query_contexts_are_never_directly_observed():
    world = build_world("E", 15)
    observed = {obs.context for obs in world.observations}
    heldout_queries = [q for q in world.queries if "heldout-context" in q.tags]
    assert heldout_queries
    assert all(q.context not in observed for q in heldout_queries)
    schedule = world.oracle["regime_schedule"]
    assert schedule[0][2] == schedule[-1][2]
    assert any(not q.answerable for q in world.queries)


def test_development_suite_integrity():
    worlds = build_suite(
        material="development-fixture",
        count=2,
        stages=STAGES,
        scale="development",
    )
    report = suite_integrity(worlds)
    assert report["passed"]
    assert report["worlds"] == 10
    assert all(report["stage_counts"][stage] == 2 for stage in STAGES)


def test_public_payload_never_contains_oracle_answers():
    world = build_world("E", 16)
    payload = world.public_payload()
    assert "oracle" not in payload
    assert all("expected_relation" not in q for q in payload["queries"])
    assert all("answerable" not in q for q in payload["queries"])
    assert all("kind" not in q for q in payload["queries"])
    assert all("tags" not in q for q in payload["queries"])
    assert all("tags" not in obs for obs in payload["observations"])
