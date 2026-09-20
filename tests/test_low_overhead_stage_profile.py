from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from low_overhead_stage_profile_campaign import (
    STAGES,
    _run_trace,
)


def test_stage_replays_preserve_prediction_checksum():
    total = _run_trace(
        0,
        18,
        timed_stage=None,
    )
    candidate = _run_trace(
        0,
        18,
        timed_stage="candidate_readout",
    )
    cache = _run_trace(
        0,
        18,
        timed_stage="cache_refresh",
    )

    assert candidate["prediction_checksum"] == total[
        "prediction_checksum"
    ]
    assert cache["prediction_checksum"] == total[
        "prediction_checksum"
    ]
    assert candidate["us_per_step"] > 0.0
    assert cache["us_per_step"] > 0.0


def test_declared_stage_set_is_complete():
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
