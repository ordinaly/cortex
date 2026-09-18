from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from dual_geometry_hardness import evaluate


def test_hardness_sweep_contains_real_degradation_regime():
    easy = np.mean(
        [
            evaluate(
                seed,
                semantic_noise=0.025,
                train_fraction=1.0,
            )["fine_rare_hit_at_1"]
            for seed in range(8)
        ]
    )
    hard = np.mean(
        [
            evaluate(
                seed,
                semantic_noise=0.35,
                train_fraction=0.125,
            )["fine_rare_hit_at_1"]
            for seed in range(8)
        ]
    )
    assert easy > 0.90
    assert hard < easy


def test_common_relation_remains_coarse_default_in_easy_regime():
    rows = [
        evaluate(
            seed,
            semantic_noise=0.025,
            train_fraction=1.0,
        )
        for seed in range(6)
    ]
    assert np.mean([row["coarse_near_hit_at_1"] for row in rows]) > 0.90
