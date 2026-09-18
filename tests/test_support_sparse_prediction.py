from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from sparse_cached_resolution import (
    PredictiveKernelCache,
    SparseCachedLocalResolutionController,
)
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
    SupportSparsePredictiveCache,
    retained_mass_support,
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


def test_retained_mass_support_has_exact_responsibility_tv():
    weights = np.asarray(
        [0.50, 0.30, 0.10, 0.06, 0.04],
        dtype=float,
    )
    support = retained_mass_support(
        weights,
        target_mass=0.90,
    )
    approx = np.zeros_like(weights)
    approx[support.indices] = support.weights
    tv = 0.5 * float(
        np.sum(np.abs(weights - approx))
    )
    assert np.isclose(tv, support.dropped_mass)
    assert support.retained_mass >= 0.90
    assert support.dropped_mass <= 0.10 + 1.0e-15


def test_full_support_bundle_matches_existing_cache():
    rng = np.random.default_rng(12)
    anchors = make_anchors(
        321,
        anchors=18,
    )
    field = _field(anchors)

    for _ in range(80):
        identity = int(
            rng.integers(0, len(anchors))
        )
        x = observe(
            anchors[identity],
            rng,
            noise=0.20,
        )
        field.observe_outcome(
            x,
            int(rng.integers(0, 3)),
        )

    baseline = PredictiveKernelCache(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0,
    )
    support_cache = SupportSparsePredictiveCache(
        field,
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0,
    )
    x = observe(
        anchors[4],
        rng,
        noise=0.20,
    )
    weights = field.perceptual_weights(x)
    indexes = np.asarray(
        [
            index % len(RESOLUTIONS)
            for index in range(len(anchors))
        ],
        dtype=int,
    )

    expected_candidates = (
        baseline.candidate_predictions(x)
    )
    expected_local = (
        baseline.predict_local_outcome(
            x,
            indexes,
        )
    )
    candidates, local, support = (
        support_cache.prediction_bundle_from_weights(
            weights,
            indexes,
            retained_mass=1.0,
        )
    )

    assert support.fraction == 1.0
    assert support.dropped_mass == 0.0
    assert np.allclose(
        candidates,
        expected_candidates,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert np.allclose(
        local,
        expected_local,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_fused_full_support_matches_v19_sparse_trace():
    rng = np.random.default_rng(13)
    anchors = make_anchors(
        654,
        anchors=18,
    )
    legacy = SparseCachedLocalResolutionController(
        _field(anchors.copy()),
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
        loss_change_tolerance=0.001,
        full_scan_interval=64,
        complexity_refresh_interval=32,
    )
    fused = FusedSupportSparseLocalResolutionController(
        _field(anchors.copy()),
        resolutions=RESOLUTIONS,
        feature_tolerance=0.0025,
        loss_change_tolerance=0.001,
        full_scan_interval=64,
        complexity_refresh_interval=32,
        retained_mass=1.0,
    )

    stable = np.asarray([0.95, 0.04, 0.01])
    divergent = np.asarray([0.01, 0.04, 0.95])

    for step in range(180):
        identity = int(
            rng.integers(0, len(anchors))
        )
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

        legacy_candidates = (
            legacy.candidate_predictions(x)
        )
        legacy_prediction, legacy_decision = (
            legacy.predict_outcome(x)
        )

        bundle = fused.prepare(x)

        assert np.allclose(
            bundle.candidate_predictions,
            legacy_candidates,
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        assert np.allclose(
            bundle.local_prediction,
            legacy_prediction,
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        assert (
            bundle.decision.resolutions
            == legacy_decision.resolutions
        )

        legacy.observe_outcome(
            x,
            outcome,
            candidate_predictions=legacy_candidates,
        )
        fused.commit(bundle, outcome)

        assert np.allclose(
            fused.field.outcome_counts,
            legacy.field.outcome_counts,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert np.allclose(
            fused.tracker.loss_numerator,
            legacy.tracker.loss_numerator,
            atol=1.0e-11,
            rtol=1.0e-11,
        )
        assert np.allclose(
            fused.tracker.anchor_mass,
            legacy.tracker.anchor_mass,
            atol=1.0e-12,
            rtol=1.0e-12,
        )


def test_support_sparse_keeps_world_model_update_exact():
    rng = np.random.default_rng(14)
    anchors = make_anchors(
        777,
        anchors=36,
    )
    exact = FusedSupportSparseLocalResolutionController(
        _field(anchors.copy()),
        resolutions=RESOLUTIONS,
        retained_mass=1.0,
    )
    sparse = FusedSupportSparseLocalResolutionController(
        _field(anchors.copy()),
        resolutions=RESOLUTIONS,
        retained_mass=0.999,
    )

    for _ in range(120):
        identity = int(
            rng.integers(0, len(anchors))
        )
        x = observe(
            anchors[identity],
            rng,
            noise=0.20,
        )
        outcome = int(
            rng.integers(0, 3)
        )

        exact_bundle = exact.prepare(x)
        sparse_bundle = sparse.prepare(x)
        exact.commit(exact_bundle, outcome)
        sparse.commit(sparse_bundle, outcome)

        assert np.allclose(
            sparse.field.outcome_counts,
            exact.field.outcome_counts,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert (
            sparse_bundle.support.dropped_mass
            <= 0.001 + 1.0e-12
        )
