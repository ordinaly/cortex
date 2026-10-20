"""Benchmark whether learned 2-simplices change local-to-global integration.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.simplicial_integration
"""
from __future__ import annotations

import json
import random
from statistics import mean

from cortex.simplicial_integration import SimplicialIntegrator
from research.five_operator_bench.higher_order_irreducibility import build_case, _feed_regime


def _measure(seed: int, *, synergistic: bool) -> dict:
    case = build_case(seed, synergistic=synergistic)
    integrator = SimplicialIntegrator(case["complex"], case["memory"])
    left_boundary = integrator.integrate(4, max_dimension=1)
    right_boundary = integrator.integrate(5, max_dimension=1)
    left_full = integrator.integrate(4, max_dimension=2)
    right_full = integrator.integrate(5, max_dimension=2)
    boundary_distance = integrator.distance(4, 5, max_dimension=1)
    full_distance = integrator.distance(4, 5, max_dimension=2)
    return {
        "edges": len(case["graph"].edges),
        "triangles": len(case["complex"].triangles),
        "boundary_distance": boundary_distance,
        "full_distance": full_distance,
        "composition_gain": max(0.0, full_distance - boundary_distance),
        "boundary_constraints": len(left_boundary.used_constraints),
        "full_constraints": len(left_full.used_constraints),
        "boundary_max_dimension": left_boundary.max_dimension,
        "full_max_dimension": left_full.max_dimension,
        "mean_boundary_entropy": mean((left_boundary.entropy, right_boundary.entropy)),
        "mean_full_entropy": mean((left_full.entropy, right_full.entropy)),
        "max_residual": max(
            left_boundary.residual,
            right_boundary.residual,
            left_full.residual,
            right_full.residual,
        ),
    }


def _aggregate(seeds, *, synergistic: bool) -> dict:
    rows = [_measure(seed, synergistic=synergistic) for seed in seeds]
    return {
        "runs": len(rows),
        "complete_boundary_rate": mean(row["edges"] == 3 for row in rows),
        "simplex_rate": mean(row["triangles"] == 1 for row in rows),
        "mean_boundary_distance": mean(row["boundary_distance"] for row in rows),
        "max_boundary_distance": max(row["boundary_distance"] for row in rows),
        "mean_full_distance": mean(row["full_distance"] for row in rows),
        "min_full_distance": min(row["full_distance"] for row in rows),
        "mean_composition_gain": mean(row["composition_gain"] for row in rows),
        "min_composition_gain": min(row["composition_gain"] for row in rows),
        "mean_entropy_reduction": mean(
            row["mean_boundary_entropy"] - row["mean_full_entropy"] for row in rows
        ),
        "max_integration_residual": max(row["max_residual"] for row in rows),
        "boundary_constraint_count": tuple(sorted({row["boundary_constraints"] for row in rows})),
        "full_constraint_count": tuple(sorted({row["full_constraints"] for row in rows})),
        "full_dimensions": tuple(sorted({row["full_max_dimension"] for row in rows})),
    }


def run_transition(seed: int = 91) -> dict:
    case = build_case(seed, synergistic=True)
    integrator = SimplicialIntegrator(case["complex"], case["memory"])
    initial_edges = len(case["graph"].edges)
    initial_triangles = len(case["complex"].triangles)
    initial_distance = integrator.distance(4, 5, max_dimension=2)

    rng = random.Random(seed + 1000)
    _feed_regime(case, rng, synergistic=False)
    reduced_edges = len(case["graph"].edges)
    reduced_triangles = len(case["complex"].triangles)
    reduced_distance = integrator.distance(4, 5, max_dimension=2)

    _feed_regime(case, rng, synergistic=True)
    restored_edges = len(case["graph"].edges)
    restored_triangles = len(case["complex"].triangles)
    restored_distance = integrator.distance(4, 5, max_dimension=2)

    return {
        "initial_edges": initial_edges,
        "initial_triangles": initial_triangles,
        "initial_distance": initial_distance,
        "reduced_edges": reduced_edges,
        "reduced_triangles": reduced_triangles,
        "reduced_distance": reduced_distance,
        "restored_edges": restored_edges,
        "restored_triangles": restored_triangles,
        "restored_distance": restored_distance,
        "passed": (
            initial_edges == reduced_edges == restored_edges == 3
            and initial_triangles == 1
            and reduced_triangles == 0
            and restored_triangles == 1
            and initial_distance > 0.60
            and reduced_distance < 0.20
            and restored_distance > 0.60
        ),
    }


def run_simplicial_integration_experiment() -> dict:
    development = tuple(range(12))
    holdout = tuple(range(12, 24))
    positive_dev = _aggregate(development, synergistic=True)
    negative_dev = _aggregate(development, synergistic=False)
    positive_holdout = _aggregate(holdout, synergistic=True)
    negative_holdout = _aggregate(holdout, synergistic=False)
    transition = run_transition()

    passed = (
        positive_dev["complete_boundary_rate"] == 1.0
        and positive_holdout["complete_boundary_rate"] == 1.0
        and negative_dev["complete_boundary_rate"] == 1.0
        and negative_holdout["complete_boundary_rate"] == 1.0
        and positive_dev["simplex_rate"] == 1.0
        and positive_holdout["simplex_rate"] == 1.0
        and negative_dev["simplex_rate"] == 0.0
        and negative_holdout["simplex_rate"] == 0.0
        and positive_holdout["max_boundary_distance"] < 0.20
        and positive_holdout["min_full_distance"] > 0.60
        and positive_holdout["min_composition_gain"] > 0.45
        and negative_holdout["mean_full_distance"] < 0.20
        and negative_holdout["mean_composition_gain"] < 1.0e-12
        and positive_holdout["boundary_constraint_count"] == (6,)
        and positive_holdout["full_constraint_count"] == (7,)
        and positive_holdout["full_dimensions"] == (2,)
        and negative_holdout["full_constraint_count"] == (6,)
        and negative_holdout["full_dimensions"] == (1,)
        and positive_holdout["max_integration_residual"] < 1.0e-7
        and negative_holdout["max_integration_residual"] < 1.0e-7
        and transition["passed"]
    )

    return {
        "version": "simplicial-integration-v1.0",
        "development_seeds": development,
        "holdout_seeds": holdout,
        "positive_development": positive_dev,
        "negative_development": negative_dev,
        "positive_holdout": positive_holdout,
        "negative_holdout": negative_holdout,
        "transition": transition,
        "passed": passed,
    }


if __name__ == "__main__":
    print(json.dumps(run_simplicial_integration_experiment(), indent=2, sort_keys=True))
