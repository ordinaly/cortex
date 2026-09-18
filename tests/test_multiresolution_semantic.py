from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from multiresolution_semantic import (
    connection_depth,
    hammer_nail_fixture,
    resolution_result,
    sqrt_probability_features,
)


def test_resolution_zero_recovers_fine_informational_geometry():
    q = [[0.8, 0.2], [0.1, 0.9]]
    phi0 = sqrt_probability_features(q)
    adjacency = [[0.0, 1.0], [1.0, 0.0]]
    result = resolution_result(phi0, adjacency, 0.0)
    expected = np.linalg.norm(phi0[0] - phi0[1]) / math.sqrt(2.0)
    assert result.distances[0, 1] == pytest.approx(expected, abs=1e-12)


def test_two_node_relation_has_exact_exponential_contraction():
    phi0 = sqrt_probability_features([[0.9, 0.1], [0.1, 0.9]])
    weight = 0.7
    adjacency = [[0.0, weight], [weight, 0.0]]
    d0 = resolution_result(phi0, adjacency, 0.0).distances[0, 1]
    for rho in [0.25, 0.5, 1.0, 2.0]:
        dr = resolution_result(phi0, adjacency, rho).distances[0, 1]
        assert dr == pytest.approx(math.exp(-2.0 * weight * rho) * d0, rel=1e-11)


def test_resolution_can_reverse_semantic_nearest_neighbor_order():
    names, phi0, adjacency = hammer_nail_fixture()
    hammer = names.index("hammer")
    nail = names.index("nail")
    screwdriver = names.index("screwdriver")

    fine = resolution_result(phi0, adjacency, 0.0).distances
    coarse = resolution_result(phi0, adjacency, 1.0).distances

    assert fine[hammer, screwdriver] < fine[hammer, nail]
    assert coarse[hammer, nail] < coarse[hammer, screwdriver]


def test_connection_depth_distinguishes_direct_relation_from_unrelated_symbol():
    names, phi0, adjacency = hammer_nail_fixture()
    hammer = names.index("hammer")
    nail = names.index("nail")
    moon = names.index("moon")

    hammer_nail = connection_depth(
        phi0, adjacency, hammer, nail, epsilon=0.10, delta=0.5, max_resolution=16
    )
    hammer_moon = connection_depth(
        phi0, adjacency, hammer, moon, epsilon=0.10, delta=0.5, max_resolution=16
    )

    assert hammer_nail is not None
    assert hammer_moon is None


def test_each_resolution_obeys_triangle_inequality():
    _names, phi0, adjacency = hammer_nail_fixture()
    for rho in [0.0, 0.25, 0.5, 1.0, 2.0]:
        distances = resolution_result(phi0, adjacency, rho).distances
        n = distances.shape[0]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert (
                        distances[i, k]
                        <= distances[i, j] + distances[j, k] + 1.0e-12
                    )
