from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from dual_geometry_ablation import make_world, run_seed


def test_all_evaluated_pairs_are_absent_from_tensor_fitting():
    _semantics, samples, heldout, _rates, _surprisal = make_world(0)
    observed = {(source, target) for source, target, _ in samples}
    assert all((source, target) not in observed for source, target, _ in heldout)


def test_latin_square_makes_rare_relation_marginals_balanced():
    _semantics, _samples, heldout, _rates, _surprisal = make_world(0)
    truths = [truth for _source, _target, truth in heldout]
    counts = [truths.count(relation) for relation in range(2, 6)]
    assert counts == [4, 4, 4, 4]


def test_full_model_beats_semantic_and_interaction_ablations():
    rows = [run_seed(seed) for seed in range(12)]
    full = np.mean(
        [row["methods"]["full-dual-geometry"]["hit_at_1"] for row in rows]
    )
    semantic = np.mean(
        [row["methods"]["semantic-only"]["hit_at_1"] for row in rows]
    )
    interaction = np.mean(
        [
            row["methods"]["interaction-marginal-only"]["hit_at_1"]
            for row in rows
        ]
    )
    assert full > 0.90
    assert semantic < 0.30
    assert interaction < 0.45


def test_rarity_resolution_is_needed_for_rare_event_forecast():
    rows = [run_seed(seed) for seed in range(12)]
    no_resolution = np.mean(
        [
            row["methods"]["tensor-no-resolution"]["hit_at_1"]
            for row in rows
        ]
    )
    full = np.mean(
        [row["methods"]["full-dual-geometry"]["hit_at_1"] for row in rows]
    )
    coarse_near = np.mean([row["coarse_near_accuracy"] for row in rows])
    fine_rare = np.mean([row["fine_rare_accuracy"] for row in rows])

    assert no_resolution < 0.20
    assert full > 0.90
    assert coarse_near > 0.90
    assert fine_rare > 0.90


def test_shuffling_semantic_structure_breaks_transfer():
    rows = [run_seed(seed) for seed in range(20)]
    full = np.mean(
        [row["methods"]["full-dual-geometry"]["hit_at_1"] for row in rows]
    )
    shuffled = np.mean(
        [
            row["methods"]["shuffled-semantic-control"]["hit_at_1"]
            for row in rows
        ]
    )
    assert full > 0.90
    assert shuffled < 0.50
