from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from exact_semantic_geometry_fast_path_campaign import (
    RESOLUTIONS,
    _field,
    run_pair_case,
)
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from support_sparse_prediction import (
    ExactSemanticGeometryPredictiveCache,
    SupportSparsePredictiveCache,
)


def test_semantic_feature_rows_are_unit_norm():
    anchors = make_anchors(1601, anchors=19)
    field = _field(anchors)
    rng = np.random.default_rng(1602)

    for _ in range(40):
        identity = int(rng.integers(0, len(anchors)))
        x = observe(anchors[identity], rng, noise=0.20)
        weights = field.perceptual_weights(x)
        field.outcome_counts *= field.evidence_decay
        field.outcome_counts[:, int(rng.integers(0, 3))] += weights

    norms = np.linalg.norm(field.features(), axis=1)
    assert np.allclose(
        norms,
        1.0,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_fast_cache_matches_legacy_sparse_refresh():
    anchors = make_anchors(1603, anchors=24)
    legacy_field = _field(anchors.copy())
    fast_field = _field(anchors.copy())

    legacy = SupportSparsePredictiveCache(
        legacy_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )
    fast = ExactSemanticGeometryPredictiveCache(
        fast_field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )

    rng = np.random.default_rng(1604)
    for _ in range(60):
        identity = int(rng.integers(0, len(anchors)))
        x = observe(anchors[identity], rng, noise=0.20)
        outcome = int(rng.integers(0, 3))

        legacy_weights = legacy_field.perceptual_weights(x)
        fast_weights = fast_field.perceptual_weights(x)
        assert np.allclose(
            legacy_weights,
            fast_weights,
            atol=1.0e-14,
            rtol=1.0e-14,
        )

        legacy_field.outcome_counts *= legacy_field.evidence_decay
        fast_field.outcome_counts *= fast_field.evidence_decay
        legacy_field.outcome_counts[:, outcome] += legacy_weights
        fast_field.outcome_counts[:, outcome] += fast_weights

        legacy_dirty = legacy.refresh()
        fast_dirty = fast.refresh()

        assert np.array_equal(legacy_dirty, fast_dirty)
        assert np.allclose(
            legacy.features,
            fast.features,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert np.allclose(
            legacy.probabilities,
            fast.probabilities,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert np.allclose(
            legacy.distances,
            fast.distances,
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        assert np.allclose(
            legacy.kernels,
            fast.kernels,
            atol=1.0e-12,
            rtol=1.0e-12,
        )


def test_short_paired_controller_trace_preserves_behavior():
    row = run_pair_case(
        0,
        18,
        equivalent_steps=8,
        divergent_steps=10,
        reconverged_steps=12,
        include_stage_attribution=False,
    )
    assert row["max_candidate_error"] <= 1.0e-12
    assert row["max_local_error"] <= 1.0e-12
    assert row["max_outcome_count_error"] <= 1.0e-12
    assert row["max_feature_error"] <= 1.0e-12
    assert row["max_probability_error"] <= 1.0e-12
    assert row["max_distance_error"] <= 1.0e-10
    assert row["max_kernel_error"] <= 1.0e-12
    assert row["dirty_disagreements"] == 0
    assert row["structural_disagreements"] == 0
