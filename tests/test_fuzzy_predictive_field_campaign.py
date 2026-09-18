from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from fuzzy_predictive_field_campaign import run_case


def test_fuzzy_reasoning_does_not_require_accurate_exact_identity():
    rows = [run_case(seed) for seed in range(6)]
    exact_identity = np.mean(
        [row["exact_identity_top1"] for row in rows]
    )
    family_mass = np.mean(
        [row["correct_family_possibility_mass"] for row in rows]
    )
    fuzzy_hit = np.mean(
        [row["fuzzy_future_hit_at_1"] for row in rows]
    )

    assert exact_identity < 0.60
    assert family_mass > 0.98
    assert fuzzy_hit > 0.80


def test_fuzzy_field_beats_hard_projection_in_sparse_world():
    rows = [run_case(seed) for seed in range(8)]
    fuzzy_hit = np.mean(
        [row["fuzzy_future_hit_at_1"] for row in rows]
    )
    hard_hit = np.mean(
        [row["hard_projection_future_hit_at_1"] for row in rows]
    )
    assert fuzzy_hit > hard_hit + 0.10


def test_predictive_complexity_tracks_roles_not_physical_anchors():
    rows = [run_case(seed) for seed in range(5)]
    rank = np.mean([row["predictive_effective_rank"] for row in rows])
    assert 6.0 < rank < 12.0


def test_fuzzy_prediction_remains_robust_as_exact_identity_drops():
    easy = [run_case(seed, perception_noise=0.08) for seed in range(5)]
    hard = [run_case(seed, perception_noise=0.25) for seed in range(5)]

    easy_identity = np.mean([row["exact_identity_top1"] for row in easy])
    hard_identity = np.mean([row["exact_identity_top1"] for row in hard])
    hard_fuzzy_prediction = np.mean(
        [row["fuzzy_future_hit_at_1"] for row in hard]
    )

    assert hard_identity < easy_identity - 0.15
    assert hard_fuzzy_prediction > 0.75
