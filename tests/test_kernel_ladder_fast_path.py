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
from kernel_ladder_fast_path_campaign import (
    run_case,
)
from support_sparse_prediction import (
    ExactSemanticGeometryPredictiveCache,
    KernelLadderSemanticGeometryPredictiveCache,
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


def test_kernel_ladder_matches_direct_kernel_rows():
    anchors = make_anchors(1801, anchors=20)
    field = _field(anchors)
    direct = ExactSemanticGeometryPredictiveCache(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )
    ladder = KernelLadderSemanticGeometryPredictiveCache(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
    )

    rng = np.random.default_rng(1802)
    distance_squared = rng.uniform(
        0.0,
        1.0,
        size=(7, 20),
    )
    expected = direct._kernel_rows_from_squared(
        distance_squared
    )
    actual = ladder._kernel_rows_from_squared(
        distance_squared
    )

    assert np.allclose(
        expected,
        actual,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert np.array_equal(
        expected == 0.0,
        actual == 0.0,
    )


def test_kernel_ladder_falls_back_for_noncanonical_resolutions():
    anchors = make_anchors(1803, anchors=12)
    field = _field(anchors)
    resolutions = (0.04, 0.08, 0.17)
    direct = ExactSemanticGeometryPredictiveCache(
        field,
        resolutions=resolutions,
        feature_tolerance=0.0025,
    )
    ladder = KernelLadderSemanticGeometryPredictiveCache(
        field,
        resolutions=resolutions,
        feature_tolerance=0.0025,
    )

    rng = np.random.default_rng(1804)
    distance_squared = rng.uniform(
        0.0,
        1.0,
        size=(5, 12),
    )
    assert np.array_equal(
        direct._kernel_rows_from_squared(
            distance_squared
        ),
        ladder._kernel_rows_from_squared(
            distance_squared
        ),
    )


def test_short_v118_trace_preserves_behavior():
    row = run_case(
        0,
        18,
        equivalent_steps=8,
        divergent_steps=10,
        reconverged_steps=12,
    )
    assert row["max_candidate_error"] <= 1.0e-12
    assert row["max_local_error"] <= 1.0e-12
    assert row["max_outcome_count_error"] <= 1.0e-12
    assert row["max_feature_error"] <= 1.0e-12
    assert row["max_probability_error"] <= 1.0e-12
    assert row["max_distance_error"] <= 1.0e-12
    assert row["max_kernel_error"] <= 1.0e-12
    assert row["underflow_mismatches"] == 0
    assert row["dirty_disagreements"] == 0
    assert row["structural_disagreements"] == 0
