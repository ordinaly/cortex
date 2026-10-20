"""Benchmark controlled causal influence over Cortex articulations.

Run with:
    PYTHONPATH=python:. python -m research.five_operator_bench.articulated_causal_influence
"""
from __future__ import annotations

import json
import random
from statistics import mean

from cortex.articulated_causal_influence import (
    ArticulatedCausalInfluence,
    CausalArticulationObservation,
)
from cortex.geometric_temporal_transport import GeometricTemporalTransport
from research.five_operator_bench.geometric_temporal_transport import _empirical_read


def _fit_effect(
    rng: random.Random,
    *,
    control_kind: str,
    control_marker: float,
    intervention_kind: str,
    intervention_marker: float,
    observations: int = 16,
    samples: int = 256,
) -> tuple[ArticulatedCausalInfluence, object, object]:
    model = ArticulatedCausalInfluence()
    first_control = None
    first_intervention = None
    for _ in range(observations):
        control = _empirical_read(
            rng,
            control_kind,
            control_marker,
            samples=samples,
            filled=control_kind != "uniform",
        )
        intervention = _empirical_read(
            rng,
            intervention_kind,
            intervention_marker,
            samples=samples,
            filled=intervention_kind != "uniform",
        )
        first_control = first_control or control
        first_intervention = first_intervention or intervention
        model.observe(CausalArticulationObservation("action", "world", False, control))
        model.observe(CausalArticulationObservation("action", "world", True, intervention))
    return model, first_control, first_intervention


def _run_seed(seed: int) -> dict:
    rng = random.Random(seed)

    # Genuine low-dimensional effect.
    marker_model, marker_control, marker_do = _fit_effect(
        rng,
        control_kind="uniform",
        control_marker=0.20,
        intervention_kind="uniform",
        intervention_marker=0.80,
    )
    marker_effect = marker_model.effect("action", "world")

    # Genuine higher-order effect: identity geometry is unchanged through dimension 1.
    parity_model, parity_control, parity_do = _fit_effect(
        rng,
        control_kind="even",
        control_marker=0.50,
        intervention_kind="odd",
        intervention_marker=0.50,
    )
    parity_effect = parity_model.effect("action", "world")
    transport = GeometricTemporalTransport()
    parity_identity_distance = transport.compare(parity_control, parity_do).distance

    # Confounded/passive association can be large even though randomized intervention is null.
    null_model, _, _ = _fit_effect(
        rng,
        control_kind="uniform",
        control_marker=0.50,
        intervention_kind="uniform",
        intervention_marker=0.50,
    )
    null_effect = null_model.effect("action", "world")

    association_model, _, _ = _fit_effect(
        rng,
        control_kind="uniform",
        control_marker=0.20,
        intervention_kind="uniform",
        intervention_marker=0.80,
    )
    observational_association = association_model.effect("action", "world").influence

    return {
        "marker_effect": marker_effect.influence,
        "marker_full_joint": marker_effect.full_joint_distance,
        "parity_effect": parity_effect.influence,
        "parity_dimension_0": dict(parity_effect.by_dimension)[0],
        "parity_dimension_1": dict(parity_effect.by_dimension)[1],
        "parity_dimension_2": dict(parity_effect.by_dimension)[2],
        "parity_full_joint": parity_effect.full_joint_distance,
        "parity_identity_distance": parity_identity_distance,
        "null_intervention_effect": null_effect.influence,
        "observational_association": observational_association,
    }


def _aggregate(seeds) -> dict:
    rows = [_run_seed(seed) for seed in seeds]
    fields = rows[0].keys()
    result = {f"mean_{field}": mean(row[field] for row in rows) for field in fields}
    result.update({f"max_{field}": max(row[field] for row in rows) for field in fields})
    result.update({f"min_{field}": min(row[field] for row in rows) for field in fields})
    result["runs"] = len(rows)
    return result


def run_articulated_causal_influence_experiment() -> dict:
    development = tuple(range(16))
    holdout = tuple(range(16, 32))
    dev = _aggregate(development)
    test = _aggregate(holdout)

    passed = (
        dev["min_marker_effect"] > 0.30
        and test["min_marker_effect"] > 0.30
        and dev["min_parity_effect"] > 0.10
        and test["min_parity_effect"] > 0.10
        and test["max_parity_identity_distance"] < 0.10
        and test["mean_parity_dimension_2"] > 0.20
        and test["max_parity_dimension_0"] < 0.05
        and test["max_parity_dimension_1"] < 0.05
        and test["max_null_intervention_effect"] < 0.05
        and test["min_observational_association"] > 0.30
        and test["mean_observational_association"] > 6.0 * test["mean_null_intervention_effect"]
    )
    return {
        "version": "articulated-causal-influence-v1.2",
        "development_seeds": development,
        "holdout_seeds": holdout,
        "development": dev,
        "holdout": test,
        "passed": passed,
    }


if __name__ == "__main__":
    print(json.dumps(run_articulated_causal_influence_experiment(), indent=2, sort_keys=True))
