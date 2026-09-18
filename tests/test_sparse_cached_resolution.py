from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution import (
    HystereticLocalResolutionController,
    local_resolution_operator,
)
from hysteretic_local_resolution_campaign import make_anchors, observe
from sparse_cached_resolution import (
    CachedLocalResolutionController,
    PredictiveKernelCache,
    SparseCachedLocalResolutionController,
)


RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)


def _field(anchors: np.ndarray) -> FuzzyPredictiveField:
    return FuzzyPredictiveField(
        anchors,
        outcomes=3,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
        evidence_decay=0.998,
    )


def test_shared_cache_matches_reference_candidate_predictions():
    rng = np.random.default_rng(4)
    anchors = make_anchors(81)
    field = _field(anchors)

    for identity in range(len(anchors)):
        for _ in range(15):
            x = observe(
                anchors[identity],
                rng,
                noise=0.20,
            )
            outcome = int(rng.integers(0, 3))
            field.observe_outcome(x, outcome)

    cache = PredictiveKernelCache(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0,
    )
    x = observe(anchors[2], rng, noise=0.20)

    expected = np.asarray(
        [
            field.predict_outcome(
                x,
                resolution=resolution,
            )
            for resolution in RESOLUTIONS
        ]
    )
    actual = cache.candidate_predictions(x)
    assert np.allclose(
        actual,
        expected,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_cached_local_operator_matches_v18_operator():
    rng = np.random.default_rng(5)
    anchors = make_anchors(91)
    field = _field(anchors)
    for _ in range(60):
        identity = int(rng.integers(0, len(anchors)))
        x = observe(
            anchors[identity],
            rng,
            noise=0.20,
        )
        field.observe_outcome(
            x,
            int(rng.integers(0, 3)),
        )

    cache = PredictiveKernelCache(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0,
    )
    indexes = np.asarray(
        [0, 1, 2, 3, 0, 1, 2, 3, 2],
        dtype=int,
    )
    resolutions = [
        RESOLUTIONS[int(index)]
        for index in indexes
    ]

    expected = local_resolution_operator(
        field,
        resolutions,
    )
    actual = cache.operator(indexes)
    assert np.allclose(
        actual,
        expected,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_exact_cached_controller_matches_v18_trace():
    rng = np.random.default_rng(6)
    anchors = make_anchors(101)
    reference = HystereticLocalResolutionController(
        _field(anchors.copy()),
        resolutions=RESOLUTIONS,
    )
    cached = CachedLocalResolutionController(
        _field(anchors.copy()),
        resolutions=RESOLUTIONS,
    )

    stable = np.asarray([0.95, 0.04, 0.01])
    divergent = np.asarray([0.01, 0.04, 0.95])

    for step in range(180):
        identity = int(rng.integers(0, len(anchors)))
        x = observe(
            anchors[identity],
            rng,
            noise=0.20,
        )
        distribution = (
            divergent
            if step >= 80 and identity == 0
            else stable
        )
        outcome = int(
            rng.choice(3, p=distribution)
        )

        ref_candidates = (
            reference.candidate_predictions(x)
        )
        ref_prediction, ref_decision = (
            reference.predict_outcome(x)
        )

        cached_candidates = (
            cached.candidate_predictions(x)
        )
        cached_prediction, cached_decision = (
            cached.predict_outcome(x)
        )

        assert np.allclose(
            cached_candidates,
            ref_candidates,
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        assert np.allclose(
            cached_prediction,
            ref_prediction,
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        assert (
            cached_decision.resolutions
            == ref_decision.resolutions
        )
        assert np.isclose(
            cached_decision.complexity,
            ref_decision.complexity,
            atol=1.0e-10,
            rtol=1.0e-10,
        )

        reference.observe_outcome(
            x,
            outcome,
            candidate_predictions=ref_candidates,
        )
        cached.observe_outcome(
            x,
            outcome,
            candidate_predictions=cached_candidates,
        )


def test_sparse_controller_exposes_bounded_diagnostics():
    rng = np.random.default_rng(7)
    anchors = make_anchors(111)
    controller = SparseCachedLocalResolutionController(
        _field(anchors),
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
        loss_change_tolerance=0.001,
        full_scan_interval=32,
        complexity_refresh_interval=16,
    )

    for _ in range(160):
        identity = int(rng.integers(0, len(anchors)))
        x = observe(
            anchors[identity],
            rng,
            noise=0.20,
        )
        candidates = controller.candidate_predictions(x)
        prediction, _ = controller.predict_outcome(x)
        assert np.isclose(prediction.sum(), 1.0)
        outcome = int(rng.integers(0, 3))
        controller.observe_outcome(
            x,
            outcome,
            candidate_predictions=candidates,
        )

    diagnostics = controller.optimization_diagnostics()
    assert 0.0 <= diagnostics["anchor_scan_fraction"] <= 1.0
    assert 0.0 <= diagnostics["svd_fraction"] <= 1.0
    assert (
        0.0
        <= diagnostics["kernel_row_refresh_fraction"]
        <= 1.0
    )
