from __future__ import annotations

from pathlib import Path

from cortex import CortexRuntime, __spec_version__, __version__, native_version

ROOT = Path(__file__).resolve().parents[1]


def test_public_versions_match_canonical_files():
    assert (ROOT / "VERSION").read_text(encoding="utf8").strip() == "1.0.0-rc.2"
    assert native_version() == "1.0.0-rc.2"
    assert __version__ == "1.0.0rc2"
    assert (ROOT / "SPEC_VERSION").read_text(encoding="utf8").strip() == "spec-v0.9.5"
    assert __spec_version__ == "spec-v0.9.5"


def test_profile_step_exposes_native_stage_attribution():
    runtime = CortexRuntime(feature_dim=8, max_entities=8, budget=4)
    read, timings = runtime.profile_step(
        [
            [0.0, 1.0, 0.2, -0.4, 0.8, 0.1, -0.7, 0.3],
            [1.0, -0.2, 0.5, 0.7, -0.3, 0.9, 0.1, -0.5],
        ],
        [(0, 1, 1)],
        0,
        [(1, 1)],
        1.0,
    )
    assert len(read.articulation_vector) == 12
    attributed = (
        timings.articulation_ns
        + timings.binding_ns
        + timings.graph_ns
        + timings.summary_ns
        + timings.memory_ns
        + timings.continual_ns
    )
    assert timings.total_ns > 0
    assert timings.total_ns >= attributed
