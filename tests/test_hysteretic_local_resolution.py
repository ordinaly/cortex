from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fuzzy_predictive_field import FuzzyPredictiveField
from hysteretic_local_resolution import (
    HystereticGlobalResolutionController,
    HystereticLocalResolutionController,
    effective_singular_rank,
    local_resolution_operator,
)
from hysteretic_local_resolution_campaign import run_seed


RESOLUTIONS = (0.03, 0.06, 0.12, 0.25)


def _field() -> FuzzyPredictiveField:
    anchors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    field = FuzzyPredictiveField(
        anchors,
        outcomes=3,
        relations=2,
        perceptual_temperature=0.10,
        semantic_resolution=0.10,
    )
    field.outcome_counts[:] = np.asarray(
        [
            [20.0, 2.0, 1.0],
            [18.0, 3.0, 1.0],
            [1.0, 2.0, 20.0],
        ]
    )
    return field


def _inject_losses(controller, losses: np.ndarray) -> None:
    controller.tracker.observations = 100
    controller.tracker.anchor_mass[:] = 10.0
    controller.tracker.loss_numerator[:] = losses * 10.0


def test_uniform_local_operator_matches_global_effective_rank():
    field = _field()
    for resolution in RESOLUTIONS:
        operator = local_resolution_operator(
            field,
            [resolution] * len(field.anchors),
        )
        local_rank = effective_singular_rank(operator)
        global_rank = field.predictive_complexity(
            resolution=resolution
        )
        assert np.isclose(local_rank, global_rank, atol=1.0e-10)


def test_global_hysteresis_sharpens_fast_and_coarsens_slowly():
    controller = HystereticGlobalResolutionController(
        _field(),
        resolutions=RESOLUTIONS,
        epsilon=0.04,
        hysteresis_margin=0.01,
        coarsen_patience=3,
        warmup_observations=1,
    )
    losses = np.asarray(
        [
            [0.10, 0.10, 0.10],
            [0.11, 0.11, 0.11],
            [0.20, 0.20, 0.20],
            [0.30, 0.30, 0.30],
        ]
    )
    _inject_losses(controller, losses)

    first = controller.decision()
    assert first.resolution == 0.06
    assert first.switch_count == 1

    again = controller.decision()
    assert again.resolution == 0.06
    assert controller.pending_count == 0

    equal = np.full((4, 3), 0.10)
    controller.tracker.loss_numerator[:] = equal * 10.0
    for _ in range(2):
        controller.tracker.observations += 1
        assert controller.decision().resolution == 0.06

    controller.tracker.observations += 1
    final = controller.decision()
    assert final.resolution == 0.25
    assert final.switch_count == 2


def test_local_hysteresis_refines_only_region_that_needs_it():
    controller = HystereticLocalResolutionController(
        _field(),
        resolutions=RESOLUTIONS,
        epsilon=0.04,
        hysteresis_margin=0.01,
        coarsen_patience=3,
        warmup_observations=1,
    )
    losses = np.asarray(
        [
            [0.10, 0.10, 0.10],
            [0.11, 0.10, 0.10],
            [0.20, 0.10, 0.10],
            [0.30, 0.10, 0.10],
        ]
    )
    _inject_losses(controller, losses)

    first = controller.decision()
    assert first.resolutions == (0.06, 0.25, 0.25)
    assert first.refined_fraction == 1.0 / 3.0
    assert first.switch_count == 1

    equal = np.full((4, 3), 0.10)
    controller.tracker.loss_numerator[:] = equal * 10.0
    for _ in range(2):
        controller.tracker.observations += 1
        assert controller.decision().resolutions[0] == 0.06

    controller.tracker.observations += 1
    final = controller.decision()
    assert final.resolutions == (0.25, 0.25, 0.25)
    assert final.refined_fraction == 0.0
    assert final.switch_count == 2


def test_campaign_smoke_is_finite_and_reports_locality():
    row = run_seed(
        0,
        equivalent_cycles=12,
        divergent_cycles=18,
        reconverged_cycles=24,
    )
    assert row["protocol"] == "hysteretic-local-resolution-v1"
    for phase in ("equivalent", "divergent", "reconverged"):
        assert np.isfinite(row[phase]["reactive"]["nll"])
        assert np.isfinite(row[phase]["hysteretic"]["nll"])
        assert np.isfinite(row[phase]["local"]["nll"])
        fraction = row[phase]["local"][
            "refined_anchor_fraction_tail"
        ]
        assert 0.0 <= fraction <= 1.0
