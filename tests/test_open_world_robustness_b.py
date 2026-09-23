from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from open_world_robustness_b_campaign import run_world
from open_world_robustness_world import build_world


def test_owr_b_characterization_exposes_temporal_metadata():
    world = build_world("B", 201)
    assert world.oracle["missing_events"] > 0
    assert world.oracle["delayed_events"] > 0
    assert world.oracle["duplicate_events"] > 0
    assert any(row.arrival_time > row.event_time for row in world.observations)
    assert any(row.duplicate_of is not None for row in world.observations)


def test_owr_b_regime_recurrence_is_explicit():
    world = build_world("B", 202)
    schedule = world.oracle["regime_schedule"]
    assert schedule[0][2] == schedule[-1][2]
    assert schedule[1][2] != schedule[0][2]
    assert schedule[2][2] not in {schedule[0][2], schedule[1][2]}


def test_owr_b_has_current_and_public_metadata_controls():
    result = run_world(build_world("B", 203))
    assert set(result["metrics"]) == {
        "current_cortex_arrival_order",
        "event_time_replay",
        "event_time_replay_deduplicated",
        "recent7_arrival_vote",
        "recent7_deduplicated_vote",
    }
    for method in result["metrics"].values():
        assert set(method) == {
            "all",
            "pre_shift",
            "regime_b",
            "regime_c",
            "recurrence_a",
        }
        for metrics in method.values():
            assert 0.0 <= metrics["coverage"] <= 1.0
            assert 0.0 <= metrics["resolved_accuracy"] <= 1.0
            assert 0.0 <= metrics["selective_risk"] <= 1.0


def test_owr_b_latency_tracks_all_noninitial_regimes():
    result = run_world(build_world("B", 204))
    assert set(result["switch_latencies"]) == {"45", "85", "125"}
