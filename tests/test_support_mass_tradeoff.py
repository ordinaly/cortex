from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from support_sparse_prediction import retained_mass_support


def test_support_size_is_monotone_in_retained_mass():
    rng = np.random.default_rng(2026)
    weights = rng.random(128)
    weights /= weights.sum()

    supports = [
        retained_mass_support(
            weights,
            target_mass=mass,
        )
        for mass in (0.99, 0.995, 0.999, 1.0)
    ]

    sizes = [len(item.indices) for item in supports]
    assert sizes == sorted(sizes)

    for mass, item in zip(
        (0.99, 0.995, 0.999, 1.0),
        supports,
    ):
        assert item.retained_mass + 1.0e-12 >= mass
        assert item.dropped_mass <= 1.0 - mass + 1.0e-12
        assert np.isclose(item.weights.sum(), 1.0)
