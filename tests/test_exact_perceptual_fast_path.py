from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from exact_perceptual_fast_path_campaign import run_pair_case
from fuzzy_predictive_field import (
    FuzzyPredictiveField,
    perceptual_possibility,
    responsibilities,
)


def test_field_fast_path_matches_generic_helper():
    rng = np.random.default_rng(991)
    anchors = rng.normal(size=(23, 11))
    observation = rng.normal(size=11)

    field = FuzzyPredictiveField(
        anchors,
        outcomes=3,
        relations=3,
        perceptual_temperature=0.07,
    )

    expected = perceptual_possibility(
        observation,
        field.anchors,
        temperature=field.perceptual_temperature,
    )
    actual = field.possibility(observation)

    assert np.allclose(
        actual,
        expected,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert np.allclose(
        field.perceptual_weights(observation),
        responsibilities(expected),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_constructor_anchor_invariant_is_unit_norm():
    rng = np.random.default_rng(992)
    field = FuzzyPredictiveField(
        rng.normal(size=(17, 9)),
        outcomes=2,
        relations=2,
    )
    norms = np.linalg.norm(field.anchors, axis=1)
    assert np.allclose(
        norms,
        1.0,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_short_paired_controller_trace_is_exact():
    row = run_pair_case(
        0,
        18,
        equivalent_steps=8,
        divergent_steps=10,
        reconverged_steps=12,
        include_stage_attribution=False,
    )
    assert row["max_weight_error"] <= 1.0e-12
    assert row["max_candidate_error"] <= 1.0e-12
    assert row["max_local_error"] <= 1.0e-12
    assert row["max_outcome_count_error"] <= 1.0e-12
    assert row["structural_disagreements"] == 0
