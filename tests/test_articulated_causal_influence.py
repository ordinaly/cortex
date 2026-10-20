from __future__ import annotations

from itertools import combinations, product
from math import log

from cortex.articulated_causal_influence import (
    ArticulatedCausalInfluence,
    CausalArticulationObservation,
)
from cortex.geometric_temporal_transport import GeometricTemporalTransport
from cortex.simplicial_integration import SimplicialIntegrationRead


PROBES = ((0,), (1,), (2,), (3,))


def _read(kind: str, marker_p: float) -> SimplicialIntegrationRead:
    distribution = {}
    for marker in (0, 1):
        pm = marker_p if marker else 1.0 - marker_p
        for bits in product((0, 1), repeat=3):
            parity = bits[0] ^ bits[1] ^ bits[2]
            if kind == "uniform":
                pb = 0.125
            elif kind == "even":
                pb = 0.25 if parity == 0 else 0.0
            elif kind == "odd":
                pb = 0.25 if parity == 1 else 0.0
            else:
                raise ValueError(kind)
            distribution[(marker, *bits)] = pm * pb
    constraints = [(i,) for i in range(4)]
    constraints.extend(combinations(range(4), 2))
    constraints.append((1, 2, 3))
    entropy = -sum(p * log(p) for p in distribution.values() if p > 0.0)
    return SimplicialIntegrationRead(
        probes=PROBES,
        distribution=tuple(sorted(distribution.items())),
        residual=0.0,
        entropy=entropy,
        used_constraints=tuple(tuple(row) for row in constraints),
        max_dimension=2,
        iterations=0,
    )


def _observe_repeated(model, *, control, intervention, n=8) -> None:
    for _ in range(n):
        model.observe(CausalArticulationObservation("action", "world", False, control))
        model.observe(CausalArticulationObservation("action", "world", True, intervention))


def test_requires_controlled_support() -> None:
    model = ArticulatedCausalInfluence()
    model.observe(CausalArticulationObservation("a", "b", True, _read("uniform", 0.8)))
    effect = model.effect("a", "b")
    assert effect.supported is False
    assert effect.influence == 0.0


def test_detects_controlled_low_dimensional_causal_shift() -> None:
    model = ArticulatedCausalInfluence()
    _observe_repeated(
        model,
        control=_read("uniform", 0.20),
        intervention=_read("uniform", 0.80),
    )
    effect = model.effect("action", "world")
    assert effect.supported is True
    assert effect.influence > 0.35
    assert effect.full_joint_distance > 0.59


def test_higher_order_causal_change_can_preserve_temporal_identity() -> None:
    control = _read("even", 0.50)
    intervention = _read("odd", 0.50)
    model = ArticulatedCausalInfluence()
    _observe_repeated(model, control=control, intervention=intervention)
    effect = model.effect("action", "world")

    transport = GeometricTemporalTransport()
    identity_distance = transport.compare(control, intervention).distance

    assert identity_distance < 1.0e-12
    assert effect.influence > 0.14
    assert dict(effect.by_dimension)[0] < 1.0e-12
    assert dict(effect.by_dimension)[1] < 1.0e-12
    assert dict(effect.by_dimension)[2] > 0.24
    assert effect.full_joint_distance > 0.99
