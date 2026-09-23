from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from open_world_robustness_d_campaign import (
    _bounded_native,
    _unbounded_temporal,
)
from open_world_robustness_world import build_world


def test_owr_d_fixture_is_sparse_salient_long_stream():
    world = build_world("D", 401)
    assert len(world.observations) == 3000
    salient = sum("salient" in row.tags for row in world.observations)
    assert salient / len(world.observations) < 0.06
    assert world.oracle["distractor_context_count"] >= 300


def test_owr_d_unbounded_temporal_reference_reaches_queries():
    report = _unbounded_temporal(build_world("D", 402))
    assert report["context_count"] > 100
    assert 0.0 <= report["coverage"] <= 1.0
    assert 0.0 <= report["resolved_accuracy"] <= 1.0


def test_owr_d_low_native_capacity_is_recorded_not_crashed():
    report = _bounded_native(build_world("D", 403), 128)
    assert report["processed_fraction"] < 1.0
    assert report["error"] is not None
    assert report["graceful_capacity_signal"] is False
    assert 0.0 <= report["temporal"]["coverage"] <= 1.0
