from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from cache_refresh_attribution_campaign import (
    ATTRIBUTION_GROUPS,
    ProfiledSupportSparsePredictiveCache,
    RESOLUTIONS,
    _field,
    run_case,
)
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from support_sparse_prediction import (
    SupportSparsePredictiveCache,
)


def test_profiled_refresh_matches_reference_exactly():
    anchors = make_anchors(801, anchors=24)
    field_ref = _field(anchors.copy())
    field_prof = _field(anchors.copy())

    reference = SupportSparsePredictiveCache(
        field_ref,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )
    profiled = ProfiledSupportSparsePredictiveCache(
        field_prof,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )

    rng = np.random.default_rng(91)
    for _ in range(40):
        identity = int(rng.integers(0, len(anchors)))
        x = observe(anchors[identity], rng, noise=0.20)
        outcome = int(rng.integers(0, 3))

        weights_ref = field_ref.perceptual_weights(x)
        weights_prof = field_prof.perceptual_weights(x)
        assert np.allclose(
            weights_ref,
            weights_prof,
            atol=1.0e-14,
            rtol=1.0e-14,
        )

        field_ref.outcome_counts *= field_ref.evidence_decay
        field_prof.outcome_counts *= field_prof.evidence_decay
        field_ref.outcome_counts[:, outcome] += weights_ref
        field_prof.outcome_counts[:, outcome] += weights_prof

        dirty_ref = reference.refresh()
        dirty_prof = profiled.refresh()

        assert np.array_equal(dirty_ref, dirty_prof)
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


def test_profile_exposes_all_engineering_groups():
    anchors = make_anchors(802, anchors=18)
    field = _field(anchors)
    cache = ProfiledSupportSparsePredictiveCache(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )

    rng = np.random.default_rng(92)
    x = observe(anchors[0], rng, noise=0.20)
    weights = field.perceptual_weights(x)
    field.outcome_counts[:, 1] += weights
    cache.refresh()

    profile = cache.profile()
    assert profile["refresh_events"] == 1
    assert 0.0 <= profile["accounted_fraction"] <= 1.0
    for group in ATTRIBUTION_GROUPS:
        assert profile[f"group_{group}_ns"] >= 0
        assert 0.0 <= profile[f"group_{group}_share"] <= 1.0


def test_cache_refresh_campaign_smoke_preserves_behavior():
    row = run_case(0, 18)
    assert row["max_candidate_error"] <= 1.0e-12
    assert row["max_local_error"] <= 1.0e-12
    assert row["max_outcome_count_error"] <= 1.0e-12
    assert row["max_feature_error"] <= 1.0e-12
    assert row["max_probability_error"] <= 1.0e-12
    assert row["max_distance_error"] <= 1.0e-12
    assert row["max_kernel_error"] <= 1.0e-12
    assert row["structural_disagreements"] == 0
    assert row["refresh_us_per_step"] > 0.0
