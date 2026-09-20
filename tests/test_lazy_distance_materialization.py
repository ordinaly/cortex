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
from lazy_distance_materialization_campaign import (
    run_case,
)
from support_sparse_prediction import (
    ExactSemanticGeometryPredictiveCache,
    LazyDistanceSemanticGeometryPredictiveCache,
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


def test_lazy_distance_view_materializes_on_access():
    anchors = make_anchors(2001, anchors=24)
    eager_field = _field(anchors.copy())
    lazy_field = _field(anchors.copy())

    eager = ExactSemanticGeometryPredictiveCache(
        eager_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )
    lazy = LazyDistanceSemanticGeometryPredictiveCache(
        lazy_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )

    assert not lazy.distance_view_stale
    assert lazy.distance_materializations == 0
    assert np.array_equal(
        eager.distances,
        lazy.distances,
    )

    rng = np.random.default_rng(2002)
    for _ in range(25):
        identity = int(rng.integers(0, len(anchors)))
        x = observe(anchors[identity], rng, noise=0.20)
        outcome = int(rng.integers(0, 3))

        eager_weights = eager_field.perceptual_weights(x)
        lazy_weights = lazy_field.perceptual_weights(x)
        assert np.allclose(
            eager_weights,
            lazy_weights,
            atol=1.0e-14,
            rtol=1.0e-14,
        )

        eager_field.outcome_counts *= eager_field.evidence_decay
        lazy_field.outcome_counts *= lazy_field.evidence_decay
        eager_field.outcome_counts[:, outcome] += eager_weights
        lazy_field.outcome_counts[:, outcome] += lazy_weights

        eager_dirty = eager.refresh()
        lazy_dirty = lazy.refresh()

        assert np.array_equal(eager_dirty, lazy_dirty)
        assert np.allclose(
            eager.kernels,
            lazy.kernels,
            atol=1.0e-12,
            rtol=1.0e-12,
        )

    assert lazy.distance_view_stale
    assert lazy.distance_materializations == 0

    actual = lazy.distances
    assert not lazy.distance_view_stale
    assert lazy.distance_materializations == 1
    assert np.allclose(
        eager.distances,
        actual,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    # Re-reading a fresh view must not trigger another rebuild.
    _ = lazy.distances
    assert lazy.distance_materializations == 1


def test_lazy_distance_refresh_full_rebuild_path():
    anchors = make_anchors(2003, anchors=12)
    eager_field = _field(anchors.copy())
    lazy_field = _field(anchors.copy())
    eager = ExactSemanticGeometryPredictiveCache(
        eager_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0,
    )
    lazy = LazyDistanceSemanticGeometryPredictiveCache(
        lazy_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0,
    )

    rng = np.random.default_rng(2004)
    identity = int(rng.integers(0, len(anchors)))
    x = observe(anchors[identity], rng, noise=0.20)
    outcome = 1
    weights = eager_field.perceptual_weights(x)
    eager_field.outcome_counts[:, outcome] += weights
    lazy_field.outcome_counts[:, outcome] += weights

    eager.refresh()
    lazy.refresh()

    assert lazy.distance_view_stale
    assert np.allclose(
        eager.kernels,
        lazy.kernels,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert np.allclose(
        eager.distances,
        lazy.distances,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_short_v120_trace_preserves_hot_behavior():
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
    assert row["hot_distance_materializations"] == 0
    assert row["stale_before_access"]
    assert not row["stale_after_access"]
    assert row["distance_materializations_after_access"] == 1
