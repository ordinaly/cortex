from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fuzzy_predictive_field import (
    FuzzyPredictiveField,
    perceptual_possibility,
)
from hysteretic_local_resolution_campaign import (
    make_anchors,
    observe,
)


def _field(anchors: np.ndarray, *, temperature: float = 0.03):
    return FuzzyPredictiveField(
        anchors,
        outcomes=3,
        relations=3,
        perceptual_temperature=temperature,
        semantic_resolution=0.04,
        evidence_decay=0.998,
    )


def test_fast_possibility_matches_generic_reference():
    rng = np.random.default_rng(41)
    anchors = make_anchors(
        1201,
        anchors=96,
    )
    field = _field(anchors)

    for _ in range(40):
        identity = int(
            rng.integers(0, len(anchors))
        )
        x = observe(
            anchors[identity],
            rng,
            noise=0.20,
        )
        fast = field.possibility(x)
        reference = perceptual_possibility(
            x,
            field.anchors,
            temperature=field.perceptual_temperature,
        )
        assert np.allclose(
            fast,
            reference,
            atol=1.0e-12,
            rtol=1.0e-12,
        )


def test_fast_weights_match_generic_reference():
    rng = np.random.default_rng(42)
    anchors = make_anchors(
        1202,
        anchors=64,
    )
    field = _field(anchors)

    for _ in range(30):
        x = observe(
            anchors[
                int(rng.integers(0, len(anchors)))
            ],
            rng,
            noise=0.20,
        )
        reference = perceptual_possibility(
            x,
            field.anchors,
            temperature=field.perceptual_temperature,
        )
        reference /= reference.sum()
        assert np.allclose(
            field.perceptual_weights(x),
            reference,
            atol=1.0e-12,
            rtol=1.0e-12,
        )


def test_fast_path_preserves_invalid_temperature_contract():
    anchors = make_anchors(
        1203,
        anchors=8,
    )
    field = _field(
        anchors,
        temperature=0.0,
    )
    with pytest.raises(ValueError):
        field.possibility(anchors[0])
