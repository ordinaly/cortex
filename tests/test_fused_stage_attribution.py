from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fused_stage_attribution_campaign import (
    candidate_from_weights,
    local_from_weights,
    run_case,
)
from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)
from support_sparse_prediction import (
    FusedSupportSparseLocalResolutionController,
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


def test_profiled_readout_matches_full_support_bundle():
    rng = np.random.default_rng(31)
    anchors = make_anchors(
        931,
        anchors=18,
    )
    controller = (
        FusedSupportSparseLocalResolutionController(
            _field(anchors),
            resolutions=RESOLUTIONS,
            retained_mass=1.0,
        )
    )

    for _ in range(70):
        identity = int(
            rng.integers(0, len(anchors))
        )
        x = observe(
            anchors[identity],
            rng,
            noise=0.20,
        )
        outcome = int(rng.integers(0, 3))
        bundle = controller.prepare(x)

        candidates = candidate_from_weights(
            controller,
            bundle.full_weights,
        )
        local = local_from_weights(
            controller,
            bundle.full_weights,
        )

        assert np.allclose(
            candidates,
            bundle.candidate_predictions,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert np.allclose(
            local,
            bundle.local_prediction,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        controller.commit(bundle, outcome)


def test_stage_attribution_smoke_is_well_formed():
    row = run_case(0, 18)
    assert row["total_us_per_step"] > 0
    assert 0.90 <= row["accounted_fraction"] <= 1.0
    shares = [
        row[key]
        for key in row
        if key.endswith("_share")
    ]
    assert shares
    assert all(
        0.0 <= value <= 1.0
        for value in shares
    )
    assert sum(shares) <= 1.0 + 1.0e-9
