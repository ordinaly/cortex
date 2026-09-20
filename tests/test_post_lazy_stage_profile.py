from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from hysteretic_local_resolution_campaign import make_anchors
from low_overhead_stage_profile_campaign import (
    STAGES,
    _controller,
)
from post_lazy_stage_profile_campaign import (
    PROTOCOL,
    _verify_lazy_default,
    run_case,
)
from support_sparse_prediction import (
    LazyDistanceSemanticGeometryPredictiveCache,
)


def test_v121_profiles_promoted_lazy_cache():
    anchors = make_anchors(
        2122,
        anchors=18,
        individual_scale=0.35,
    )
    controller = _controller(anchors)
    assert isinstance(
        controller.cache,
        LazyDistanceSemanticGeometryPredictiveCache,
    )
    _verify_lazy_default()


def test_v121_reuses_v119_stage_boundaries():
    assert STAGES == (
        "hysteresis_scan",
        "complexity",
        "perceptual",
        "candidate_readout",
        "local_readout",
        "tracker_update",
        "field_update",
        "cache_refresh",
    )


def test_v121_case_preserves_replay_semantics():
    row = run_case(10, 18)
    assert row["protocol"] == PROTOCOL
    assert row["lazy_cache_default"]
    assert row["checksum_error"] <= 1.0e-12
    assert row["total_us_per_step"] > 0.0
