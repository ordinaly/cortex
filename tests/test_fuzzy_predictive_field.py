from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fuzzy_predictive_field import (
    FuzzyPredictiveField,
    effective_rank,
    perceptual_possibility,
    predictive_kernel,
    responsibilities,
    semantic_features,
)


def test_possibility_is_not_required_to_sum_to_one():
    anchors = np.eye(3)
    possibility = perceptual_possibility(
        [1.0, 0.0, 0.0],
        anchors,
        temperature=0.2,
    )
    assert possibility[0] == 1.0
    assert float(possibility.sum()) != 1.0
    assert abs(float(responsibilities(possibility).sum()) - 1.0) < 1e-12


def test_predictively_equivalent_anchors_have_low_effective_rank():
    counts = np.asarray(
        [
            [90.0, 10.0, 0.0],
            [91.0, 9.0, 0.0],
            [89.0, 11.0, 0.0],
            [0.0, 10.0, 90.0],
            [0.0, 9.0, 91.0],
            [0.0, 11.0, 89.0],
        ]
    )
    features = semantic_features(counts)
    kernel = predictive_kernel(features, resolution=0.10)
    rank = effective_rank(kernel)
    assert rank < 3.0


def test_predictive_divergence_continuously_reduces_affinity():
    anchors = np.eye(3)
    field = FuzzyPredictiveField(
        anchors,
        outcomes=3,
        relations=3,
        semantic_resolution=0.10,
    )
    field.outcome_counts[:] = np.asarray(
        [
            [80.0, 20.0, 0.0],
            [80.0, 20.0, 0.0],
            [80.0, 20.0, 0.0],
        ]
    )
    before = field.kernel()[0, 1]

    field.outcome_counts[0] = [5.0, 10.0, 85.0]
    after = field.kernel()[0, 1]

    assert before > 0.99
    assert after < 0.10
