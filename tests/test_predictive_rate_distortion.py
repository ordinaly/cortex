from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fuzzy_predictive_field import FuzzyPredictiveField
from predictive_rate_distortion import AdaptiveResolutionController
from predictive_rate_distortion_campaign import make_anchors, run_seed


def test_controller_selects_minimum_complexity_admissible_resolution():
    anchors = make_anchors(1)
    field = FuzzyPredictiveField(
        anchors,
        outcomes=3,
        relations=3,
        perceptual_temperature=0.03,
        semantic_resolution=0.04,
    )
    controller = AdaptiveResolutionController(
        field,
        resolutions=(0.03, 0.06, 0.12, 0.25),
        epsilon=0.05,
        warmup_observations=1,
    )

    controller.observations = 100
    controller.anchor_mass[:] = 10.0
    # Candidate 0 is best; candidates 1 and 2 remain within epsilon.
    # Candidate 2 should be chosen if it has the lowest effective rank.
    controller.loss_numerator[:] = np.asarray(
        [
            [2.00, 2.00, 2.00],
            [2.20, 2.20, 2.20],
            [2.40, 2.40, 2.40],
            [4.00, 4.00, 4.00],
        ]
    )
    # Divide by anchor mass => [0.20, 0.22, 0.24, 0.40].
    decision = controller.decision()
    assert decision.distortion <= decision.best_distortion + controller.epsilon + 1e-12
    assert decision.resolution in decision.admissible


def test_controller_sharpens_then_recoarsens():
    rows = [
        run_seed(
            seed,
            equivalent_cycles=350,
            divergent_cycles=550,
            reconverged_cycles=700,
        )
        for seed in range(5)
    ]

    equivalent = np.mean(
        [row["equivalent"]["mean_resolution_tail"] for row in rows]
    )
    divergent = np.mean(
        [row["divergent"]["mean_resolution_tail"] for row in rows]
    )
    reconverged = np.mean(
        [row["reconverged"]["mean_resolution_tail"] for row in rows]
    )

    assert divergent < equivalent - 0.04
    assert reconverged > divergent + 0.04


def test_adaptive_resolution_preserves_divergent_prediction_quality():
    rows = [
        run_seed(
            seed,
            equivalent_cycles=350,
            divergent_cycles=550,
            reconverged_cycles=500,
        )
        for seed in range(5)
    ]

    adaptive = np.mean(
        [row["divergent"]["adaptive_nll"] for row in rows]
    )
    coarse = np.mean(
        [row["divergent"]["coarse_nll"] for row in rows]
    )
    fine = np.mean(
        [row["divergent"]["fine_nll"] for row in rows]
    )

    assert adaptive < coarse - 0.03
    assert adaptive <= fine + 0.08


def test_complexity_tracks_predictive_need():
    rows = [
        run_seed(
            seed,
            equivalent_cycles=350,
            divergent_cycles=550,
            reconverged_cycles=700,
        )
        for seed in range(5)
    ]

    equivalent = np.mean(
        [row["equivalent"]["mean_complexity_tail"] for row in rows]
    )
    divergent = np.mean(
        [row["divergent"]["mean_complexity_tail"] for row in rows]
    )
    reconverged = np.mean(
        [row["reconverged"]["mean_complexity_tail"] for row in rows]
    )

    assert divergent > equivalent + 0.25
    assert reconverged < divergent - 0.20
