from __future__ import annotations

import random

from cortex.simplicial_integration import SimplicialIntegrator
from research.five_operator_bench.higher_order_irreducibility import build_case, _feed_regime


def test_filled_simplex_changes_articulation_beyond_same_boundary() -> None:
    case = build_case(14, synergistic=True)
    assert len(case["graph"].edges) == 3
    assert len(case["complex"].triangles) == 1

    integrator = SimplicialIntegrator(case["complex"], case["memory"])
    boundary = integrator.distance(4, 5, max_dimension=1)
    filled = integrator.distance(4, 5, max_dimension=2)
    left_boundary = integrator.integrate(4, max_dimension=1)
    left_filled = integrator.integrate(4, max_dimension=2)

    assert boundary < 0.20
    assert filled > 0.60
    assert filled - boundary > 0.45
    assert left_boundary.max_dimension == 1
    assert left_filled.max_dimension == 2
    assert len(left_boundary.used_constraints) == 6
    assert len(left_filled.used_constraints) == 7
    assert left_boundary.residual < 1.0e-7
    assert left_filled.residual < 1.0e-7


def test_reducible_k3_does_not_gain_a_fake_collective_articulation() -> None:
    case = build_case(17, synergistic=False)
    assert len(case["graph"].edges) == 3
    assert len(case["complex"].triangles) == 0

    integrator = SimplicialIntegrator(case["complex"], case["memory"])
    boundary = integrator.distance(4, 5, max_dimension=1)
    full = integrator.distance(4, 5, max_dimension=2)
    full_read = integrator.integrate(4, max_dimension=2)

    assert boundary < 0.20
    assert abs(full - boundary) < 1.0e-12
    assert integrator.composition_gain(4, 5) < 1.0e-12
    assert full_read.max_dimension == 1
    assert len(full_read.used_constraints) == 6


def test_integration_tracks_dimension_two_transition_without_boundary_change() -> None:
    case = build_case(91, synergistic=True)
    integrator = SimplicialIntegrator(case["complex"], case["memory"])
    initial = integrator.distance(4, 5, max_dimension=2)
    assert len(case["graph"].edges) == 3
    assert len(case["complex"].triangles) == 1

    rng = random.Random(1091)
    _feed_regime(case, rng, synergistic=False)
    reduced = integrator.distance(4, 5, max_dimension=2)
    assert len(case["graph"].edges) == 3
    assert len(case["complex"].triangles) == 0

    _feed_regime(case, rng, synergistic=True)
    restored = integrator.distance(4, 5, max_dimension=2)
    assert len(case["graph"].edges) == 3
    assert len(case["complex"].triangles) == 1

    assert initial > 0.60
    assert reduced < 0.20
    assert restored > 0.60
