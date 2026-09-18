from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from adaptive_predictive_distinction import run_seed


def test_prediction_creates_distinction_without_hard_split():
    rows = [run_seed(seed, post_cycles=400) for seed in range(6)]

    exact_identity = np.mean(
        [row["exact_identity_top1"] for row in rows]
    )
    before_affinity = np.mean(
        [row["before_changed_to_stable_affinity"] for row in rows]
    )
    after_affinity = np.mean(
        [row["after_changed_to_stable_affinity"] for row in rows]
    )
    before_rank = np.mean(
        [row["before_effective_rank"] for row in rows]
    )
    after_rank = np.mean(
        [row["after_effective_rank"] for row in rows]
    )

    assert exact_identity < 0.75
    assert before_affinity > 0.75
    assert after_affinity < 0.20
    assert after_rank > before_rank + 0.60


def test_future_distribution_adapts_continuously():
    rows = [run_seed(seed, post_cycles=800) for seed in range(6)]

    before = np.mean(
        [row["before_changed_future_probability"] for row in rows]
    )
    after = np.mean(
        [row["after_changed_future_probability"] for row in rows]
    )
    stable = np.mean(
        [row["after_stable_future_probability"] for row in rows]
    )

    assert before < 0.05
    assert after > before + 0.30
    assert stable > 0.60


def test_unchanged_anchors_remain_mutually_close():
    rows = [run_seed(seed, post_cycles=800) for seed in range(6)]
    stable_affinity = np.mean(
        [row["after_stable_to_stable_affinity"] for row in rows]
    )
    changed_affinity = np.mean(
        [row["after_changed_to_stable_affinity"] for row in rows]
    )

    assert stable_affinity > changed_affinity + 0.40


def test_resolution_controls_predictive_distinction_strength():
    rows = [run_seed(seed, post_cycles=800) for seed in range(6)]

    fine_affinity = np.mean(
        [
            row["resolution_sweep"]["0.03"][
                "changed_to_stable_affinity"
            ]
            for row in rows
        ]
    )
    coarse_affinity = np.mean(
        [
            row["resolution_sweep"]["0.25"][
                "changed_to_stable_affinity"
            ]
            for row in rows
        ]
    )
    fine_rank = np.mean(
        [
            row["resolution_sweep"]["0.03"]["effective_rank"]
            for row in rows
        ]
    )
    coarse_rank = np.mean(
        [
            row["resolution_sweep"]["0.25"]["effective_rank"]
            for row in rows
        ]
    )
    fine_changed_future = np.mean(
        [
            row["resolution_sweep"]["0.03"][
                "changed_future_probability"
            ]
            for row in rows
        ]
    )
    coarse_changed_future = np.mean(
        [
            row["resolution_sweep"]["0.25"][
                "changed_future_probability"
            ]
            for row in rows
        ]
    )

    assert fine_affinity < coarse_affinity - 0.40
    assert fine_rank > coarse_rank + 0.40
    assert fine_changed_future > coarse_changed_future + 0.05
