from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fused_stage_attribution_campaign import (
    RESOLUTIONS,
)
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from optimized_cache_refresh_attribution_campaign import (
    GROUPS,
    ProfiledExactSemanticGeometryCache,
    _field,
    run_case,
)
from support_sparse_prediction import (
    ExactSemanticGeometryPredictiveCache,
)


def test_profiled_optimized_cache_matches_reference():
    anchors = make_anchors(
        1701,
        anchors=24,
    )
    reference_field = _field(
        anchors.copy()
    )
    profiled_field = _field(
        anchors.copy()
    )

    reference = ExactSemanticGeometryPredictiveCache(
        reference_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )
    profiled = ProfiledExactSemanticGeometryCache(
        profiled_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )

    rng = np.random.default_rng(1702)
    for _ in range(50):
        identity = int(
            rng.integers(
                0,
                len(anchors),
            )
        )
        observation = observe(
            anchors[identity],
            rng,
            noise=0.20,
        )
        outcome = int(
            rng.integers(0, 3)
        )

        reference_weights = (
            reference_field.perceptual_weights(
                observation
            )
        )
        profiled_weights = (
            profiled_field.perceptual_weights(
                observation
            )
        )
        assert np.allclose(
            reference_weights,
            profiled_weights,
            atol=1.0e-14,
            rtol=1.0e-14,
        )

        reference_field.outcome_counts *= (
            reference_field.evidence_decay
        )
        profiled_field.outcome_counts *= (
            profiled_field.evidence_decay
        )
        reference_field.outcome_counts[
            :, outcome
        ] += reference_weights
        profiled_field.outcome_counts[
            :, outcome
        ] += profiled_weights

        reference_dirty = reference.refresh()
        profiled_dirty = profiled.refresh()

        assert np.array_equal(
            reference_dirty,
            profiled_dirty,
        )
        assert np.allclose(
            reference.features,
            profiled.features,
            atol=1.0e-14,
            rtol=1.0e-14,
        )
        assert np.allclose(
            reference.probabilities,
            profiled.probabilities,
            atol=1.0e-14,
            rtol=1.0e-14,
        )
        assert np.allclose(
            reference.distances,
            profiled.distances,
            atol=1.0e-14,
            rtol=1.0e-14,
        )
        assert np.allclose(
            reference.kernels,
            profiled.kernels,
            atol=1.0e-14,
            rtol=1.0e-14,
        )


def test_profile_exposes_optimized_groups():
    anchors = make_anchors(
        1703,
        anchors=18,
    )
    field = _field(anchors)
    cache = ProfiledExactSemanticGeometryCache(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )
    rng = np.random.default_rng(1704)

    observation = observe(
        anchors[0],
        rng,
        noise=0.20,
    )
    weights = field.perceptual_weights(
        observation
    )
    field.outcome_counts[:, 1] += weights
    cache.refresh()

    profile = cache.profile()
    assert profile["refresh_events"] == 1
    assert 0.0 <= profile[
        "accounted_fraction"
    ] <= 1.0
    for group in GROUPS:
        assert profile[
            f"group_{group}_ns"
        ] >= 0
        assert 0.0 <= profile[
            f"group_{group}_share"
        ] <= 1.0


def test_optimized_cache_attribution_smoke():
    row = run_case(0, 18)
    assert row["max_candidate_error"] <= 1.0e-12
    assert row["max_local_error"] <= 1.0e-12
    assert row["max_outcome_count_error"] <= 1.0e-12
    assert row["max_feature_error"] <= 1.0e-12
    assert row["max_probability_error"] <= 1.0e-12
    assert row["max_distance_error"] <= 1.0e-12
    assert row["max_kernel_error"] <= 1.0e-12
    assert row["dirty_disagreements"] == 0
    assert row["structural_disagreements"] == 0
    assert row["refresh_us_per_step"] > 0.0
