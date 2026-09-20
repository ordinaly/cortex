from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fused_stage_attribution_campaign import RESOLUTIONS
from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from single_buffer_kernel_refresh_campaign import run_case
from support_sparse_prediction import (
    LazyDistanceSemanticGeometryPredictiveCache,
    SingleBufferKernelRefreshPredictiveCache,
)


def _field(anchors):
    return FuzzyPredictiveField(
        anchors,
        outcomes=3,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
        evidence_decay=0.998,
    )


def _update(field, x, outcome):
    weights = field.perceptual_weights(x)
    field.outcome_counts *= field.evidence_decay
    field.outcome_counts[:, outcome] += weights


def test_single_buffer_partial_refresh_matches_lazy_reference():
    anchors = make_anchors(2201, anchors=24)
    reference_field = _field(anchors.copy())
    candidate_field = _field(anchors.copy())

    reference = LazyDistanceSemanticGeometryPredictiveCache(
        reference_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )
    candidate = SingleBufferKernelRefreshPredictiveCache(
        candidate_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )

    assert candidate._kernel_scratch.ndim == 2
    assert candidate.scratch_capacity_rows == 6
    assert candidate.single_buffer_refreshes == 0

    rng = np.random.default_rng(2202)
    for _ in range(30):
        identity = int(rng.integers(0, len(anchors)))
        x = observe(anchors[identity], rng, noise=0.20)
        outcome = int(rng.integers(0, 3))

        _update(reference_field, x, outcome)
        _update(candidate_field, x, outcome)

        reference_dirty = reference.refresh()
        candidate_dirty = candidate.refresh()

        assert np.array_equal(
            reference_dirty,
            candidate_dirty,
        )
        assert np.allclose(
            reference.features,
            candidate.features,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert np.allclose(
            reference.probabilities,
            candidate.probabilities,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert np.allclose(
            reference.kernels,
            candidate.kernels,
            atol=1.0e-12,
            rtol=1.0e-12,
        )

    assert candidate.single_buffer_refreshes > 0
    assert reference.distance_materializations == 0
    assert candidate.distance_materializations == 0
    assert reference.distance_view_stale
    assert candidate.distance_view_stale
    assert np.allclose(
        reference.distances,
        candidate.distances,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_single_buffer_full_rebuild_does_not_use_3d_kernel_builder():
    anchors = make_anchors(2203, anchors=12)
    reference_field = _field(anchors.copy())
    candidate_field = _field(anchors.copy())

    reference = LazyDistanceSemanticGeometryPredictiveCache(
        reference_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0,
    )
    candidate = SingleBufferKernelRefreshPredictiveCache(
        candidate_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0,
    )

    def forbidden(_distance_squared):
        raise AssertionError(
            "v1.22 refresh must not call the 3-D kernel builder"
        )

    candidate._kernel_rows_from_squared = forbidden

    rng = np.random.default_rng(2204)
    identity = int(rng.integers(0, len(anchors)))
    x = observe(anchors[identity], rng, noise=0.20)
    outcome = 1

    _update(reference_field, x, outcome)
    _update(candidate_field, x, outcome)

    reference_dirty = reference.refresh()
    candidate_dirty = candidate.refresh()

    assert np.array_equal(reference_dirty, candidate_dirty)
    assert candidate.scratch_capacity_rows == len(anchors)
    assert candidate.scratch_reallocations >= 1
    assert candidate.single_buffer_refreshes == 1
    assert np.allclose(
        reference.kernels,
        candidate.kernels,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_short_v122_trace_preserves_behavior():
    row = run_case(0, 18)
    assert row["max_candidate_error"] <= 1.0e-12
    assert row["max_local_error"] <= 1.0e-12
    assert row["max_outcome_count_error"] <= 1.0e-12
    assert row["max_feature_error"] <= 1.0e-12
    assert row["max_probability_error"] <= 1.0e-12
    assert row["max_kernel_error"] <= 1.0e-12
    assert row["max_distance_error_after_access"] <= 1.0e-12
    assert row["dirty_disagreements"] == 0
    assert row["structural_disagreements"] == 0
    assert row["reference_hot_distance_materializations"] == 0
    assert row["candidate_hot_distance_materializations"] == 0
    assert row["candidate_scratch_ndim"] == 2
    assert row["candidate_single_buffer_refreshes"] > 0
