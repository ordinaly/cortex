from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from dual_geometry_tensor import (
    fit_tensor_bridge,
    interaction_distance,
    relation_surprisal,
    resolved_event_scores,
    toy_fixture,
)


def test_interaction_resolution_separates_rare_profiles():
    surprisal = relation_surprisal([1000.0, 180.0, 20.0])
    a = [0.90, 0.09, 0.01]
    b = [0.90, 0.01, 0.09]

    coarse = interaction_distance(
        a, b, surprisal, eta=float(surprisal[0]), bandwidth=0.30
    )
    fine = interaction_distance(
        a, b, surprisal, eta=float(surprisal[2]), bandwidth=0.30
    )
    assert fine > 5.0 * coarse


def test_tensor_bridge_generalizes_to_unseen_semantic_pair():
    names, semantics, _counts, surprisal, samples, heldout = toy_fixture()
    assert all((source, target) != heldout for source, target, _ in samples)

    bridge = fit_tensor_bridge(semantics, samples)
    source, target = heldout
    prediction = bridge.predict(semantics[source], semantics[target])

    # The bridge should infer the rare unlock relation for the unseen
    # key-like -> lock-like semantic configuration.
    assert prediction[2] > prediction[1]
    assert prediction[2] > prediction[0]
    assert names[source] == "key-b"
    assert names[target] == "lock-b"


def test_interaction_resolution_changes_ranked_event_type():
    _names, semantics, _counts, surprisal, samples, heldout = toy_fixture()
    bridge = fit_tensor_bridge(semantics, samples)
    source, target = heldout

    coarse = resolved_event_scores(
        bridge,
        semantics[source],
        semantics[target],
        surprisal,
        eta=float(surprisal[0]),
        bandwidth=0.35,
    )
    fine = resolved_event_scores(
        bridge,
        semantics[source],
        semantics[target],
        surprisal,
        eta=float(surprisal[2]),
        bandwidth=0.35,
    )

    assert int(np.argmax(coarse)) == 0  # near
    assert int(np.argmax(fine)) == 2    # unlock


def test_direct_confidence_overrides_bridge_when_pair_is_well_observed():
    _names, semantics, _counts, surprisal, samples, heldout = toy_fixture()
    bridge = fit_tensor_bridge(semantics, samples)
    source, target = heldout

    scores = resolved_event_scores(
        bridge,
        semantics[source],
        semantics[target],
        surprisal,
        eta=float(surprisal[1]),
        direct_rates=[0.05, 0.95, 0.01],
        direct_confidence=[1.0, 1.0, 1.0],
        bandwidth=0.35,
    )
    assert int(np.argmax(scores)) == 1
