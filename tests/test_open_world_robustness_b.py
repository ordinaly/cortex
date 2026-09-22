from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from open_world_robustness_b_campaign import run_world
from open_world_robustness_world import build_world


def test_owr_b_queries_cover_transition_phases_and_recurrence():
    world = build_world("B", 201)
    phases = {tag for query in world.queries for tag in query.tags}
    assert {"boundary", "early", "settled", "stable", "recurrence"} <= phases
    times = {query.query_time for query in world.queries}
    assert {45, 48, 55, 85, 88, 95, 125, 128, 135, 160} <= times


def test_owr_b_public_stream_exposes_time_and_duplicate_provenance():
    world = build_world("B", 202)
    assert world.oracle["learner_receives_event_time"] is True
    assert world.oracle["learner_receives_duplicate_provenance"] is True
    assert any(obs.arrival_time > obs.event_time for obs in world.observations)
    duplicates = [obs for obs in world.observations if obs.duplicate_of is not None]
    assert duplicates
    assert all(obs.duplicate_of for obs in duplicates)


def test_owr_b_has_missingness_and_recurrent_relation():
    world = build_world("B", 203)
    assert world.oracle["missing_events"] > 0
    schedule = world.oracle["regime_schedule"]
    assert schedule[0][2] == schedule[-1][2]
    assert schedule[0][2] != schedule[1][2]
    assert schedule[1][2] != schedule[2][2]


def test_owr_b_characterization_declares_temporal_baselines():
    result = run_world(build_world("B", 204))
    assert set(result["metrics"]) == {
        "current_cortex_arrival_order",
        "current_cortex_public_dedupe",
        "arrival_recent_five_vote",
        "event_time_recent_five_vote",
        "oracle_current_truth",
    }
    partitions = {
        "all",
        "boundary",
        "early",
        "settled",
        "stable",
        "recurrence",
        "novel_regime",
    }
    for method in result["metrics"].values():
        assert set(method) == partitions
        for metrics in method.values():
            assert 0.0 <= metrics["coverage"] <= 1.0
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert 0.0 <= metrics["resolved_accuracy"] <= 1.0
            assert 0.0 <= metrics["selective_risk"] <= 1.0


def test_owr_b_latency_tracks_novel_and_recurrent_transitions():
    result = run_world(build_world("B", 205))
    for method in (
        "current_cortex_arrival_order",
        "current_cortex_public_dedupe",
    ):
        assert set(result["latency"][method]) == {"all", "novel", "recurrence"}
        assert result["latency"][method]["novel"]["cases"] == 16
        assert result["latency"][method]["recurrence"]["cases"] == 8
